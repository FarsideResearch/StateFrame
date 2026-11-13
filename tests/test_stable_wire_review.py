"""Final-review regressions for opaque rows, frame order and legacy resume."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

import kernel_stable_wire as stable
from test_newest_user_turn_survives_wire_projection import ADVISORY, _final_cache_shaped_body
from test_stable_wire import Journal, Meter, _invoke, _window


QUOTED_SOURCE = "[Current source: pkg/a.py] quoted by the model"


@pytest.mark.parametrize("content", [QUOTED_SOURCE, _window()["content"]],
                         ids=["unparseable", "exact-window-copy"])
@pytest.mark.parametrize("where", ["assistant", "tool", "captured", "newest", "native_user",
                                  "pre_frame_assistant", "pre_frame_tool"])
def test_source_shaped_unowned_rows_are_opaque(monkeypatch, content, where):
    body = _final_cache_shaped_body()
    if where in {"assistant", "tool"}:
        row = next(r for r in body["messages"] if r.get("role") == where)
    elif where.startswith("pre_frame_"):
        row = {"role": where.removeprefix("pre_frame_"), "content": content}
        body["messages"].insert(stable._frame_index(body["messages"]), row)
    elif where == "native_user":
        row = {"role": "user", "content": content}
        body["messages"].insert(len(body["messages"]) - 1, row)
    elif where == "captured":
        row = next(r for r in body["messages"] if r.get("content") == ADVISORY)
    else:
        row = body["messages"][-1]
    row["content"] = content
    captured = (copy.deepcopy(row),) if where == "captured" else ()
    capture = SimpleNamespace(journal=Journal(), request_id="opaque-source",
                              _framefit5_newest_user_suffix=captured)
    # Install the deterministic meter/source-proof fixtures, then exercise the
    # actual projection and all serialized readback paths with captured provenance.
    _invoke(monkeypatch, _final_cache_shaped_body(), SimpleNamespace(_stable_wire_state=None))
    result = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    recorded = json.loads(json.dumps(result))["messages"]
    assert stable.recorded_window_contents(recorded) == [_window()["content"]]
    canonical, _ = stable.canonicalize_stable_wire(recorded)
    assert row in canonical
    assert stable._order_preserved(body["messages"], recorded)
    if where == "newest":
        assert canonical[-1] == row
    assert stable._window_entries(recorded) == []  # No out-of-band ownership.


@pytest.mark.parametrize("role", ["assistant", "tool", "user"])
@pytest.mark.parametrize("content", [QUOTED_SOURCE, _window()["content"]],
                         ids=["unparseable", "exact-window-copy"])
def test_window_inventory_without_ownership_never_parses_rows(role, content):
    assert stable._window_entries([{"role": role, "content": content}]) == []


def test_plain_order_proof_without_ownership_keeps_source_shaped_rows_opaque(monkeypatch):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    body = _final_cache_shaped_body()
    for row in body["messages"]:
        if row.get("role") in {"assistant", "tool"} or row.get("content") == ADVISORY:
            row["content"] = QUOTED_SOURCE
    assert stable._order_preserved(body["messages"], copy.deepcopy(body["messages"]))


@pytest.mark.parametrize("role", ["assistant", "tool", "user"])
def test_post_frame_source_text_is_opaque_even_in_host_owned_tail(monkeypatch, role):
    body = _final_cache_shaped_body()
    frame = stable._frame_index(body["messages"])
    row = {"role": role, "content": QUOTED_SOURCE}
    body["messages"].insert(frame + 1, row)
    # Explicit exchange boundary leaves the extra post-frame row host-owned.
    # It still cannot become a source window or reach the source parser.
    from kernel_stable_wire_ownership import request_ownership
    ownership = request_ownership(body["messages"], frame, exchange_start=frame + 2)
    assert [w["content"] for w in stable._window_entries(body["messages"], ownership)] == [
        _window()["content"]]
    result, capture, _ = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert result == body
    assert capture._final_cache_layout_record["stable_wire_reason"] == "projection_moves_frame"


def test_frame_crossing_a_pre_frame_row_fails_open(monkeypatch):
    body = _final_cache_shaped_body()
    # The old proof exempts the frame, accepting this reordered plain view.
    moved = copy.deepcopy(body["messages"])
    frame = stable._frame_index(moved)
    moved[frame - 1], moved[frame] = moved[frame], moved[frame - 1]
    assert not stable._order_preserved(body["messages"], moved)

    project = stable._project_body

    def move_frame(*args, **kwargs):
        projected, status, frame_text = project(*args, **kwargs)
        from kernel_stable_wire_ownership import ProjectionOrder, projection_order
        rows = projected["messages"]
        at = stable._frame_index(rows)
        order = projection_order(rows, at)
        # Reclassify the last pre-frame row as exchange: the canonical frame
        # crosses it, while every other row retains its relative order.
        old = order.encode()
        new = ProjectionOrder(order.prefix_end, order.tail_count - 1,
                              order.notices, order.newest_count).encode()
        rows[at + 2]["content"] = rows[at + 2]["content"].removesuffix(old) + new
        return projected, status, frame_text

    monkeypatch.setattr(stable, "_project_body", move_frame)
    result, capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert result == body
    assert capture._final_cache_layout_record["stable_wire_reason"] == "projection_moves_frame"
    assert any(e.get("reason") == "projection_moves_frame" for e in journal.events)


@pytest.mark.parametrize("switch", ["on", "off"])
def test_resume_legacy_recording_without_ownership_fails_open(monkeypatch, tmp_path, switch):
    import kernel_bounded_http
    import kernel_cache_layout_candidate as candidate
    from kernel_stable_wire_ownership import ORDER_MARKER
    from stateframe import _child
    from stateframe.replay_tape import Replayer

    body = _final_cache_shaped_body()
    result, _, _ = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    legacy = copy.deepcopy(result)
    legacy["messages"][stable._frame_index(legacy["messages"]) + 2]["content"] = (
        legacy["messages"][stable._frame_index(legacy["messages"]) + 2]["content"]
        .split(ORDER_MARKER, 1)[0])
    request = tmp_path / "legacy.request.json"
    response = tmp_path / "legacy.response.json"
    request.write_text(json.dumps(legacy))
    response.write_bytes(b"{}")
    replay = Replayer([{"send": 1, "request": str(request), "response": str(response),
                        "response_sha256": hashlib.sha256(b"{}").hexdigest()}])
    for attr in ("_REPLAY_LAYOUT_DECISION_PROVIDER", "_REPLAY_STABLE_WIRE_DECISION_PROVIDER",
                 "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER"):
        monkeypatch.setattr(candidate, attr, None)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", kernel_bounded_http.request_bytes)
    run = tmp_path / "resume"
    run.mkdir()
    _child.install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions", replay)
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", switch)
    capture = SimpleNamespace(journal=Journal(), request_id="legacy-resume")
    runtime = SimpleNamespace(_stable_wire_state=None)
    rebuilt = stable.project_stable_wire(
        capture, runtime, body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    assert rebuilt == body
    assert runtime._stable_wire_state is None
    assert capture._final_cache_layout_record["stable_wire_reason"] == "recorded_row_ownership_unavailable"
    # The changed wire cannot borrow an old response. Replay diverges normally
    # and continues live from the verified prefix instead of crashing.
    assert replay.answer(json.dumps(rebuilt).encode()) is None
