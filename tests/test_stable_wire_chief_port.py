"""Chief 6673c58e/a44966cf regressions adapted to captured ownership and epoch APIs."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_cache_layout_candidate as candidate
import kernel_stable_wire as stable
from kernel_stable_wire_ownership import ProjectionOrder, projection_order, request_ownership
from test_newest_user_turn_survives_wire_projection import (
    ADVISORY, SUMMARY, _final_cache_shaped_body,
)
from test_stable_wire import (
    ForcedRelayoutMeter, _body, _events, _invoke, _recorded_k2_sends, _replay_with, _window,
)
from test_stable_wire_fit import (
    SizeMeter, UnderTrimmer, _failed_open_events, _inputs, _layout_body, _render,
)


def _ownership(rows):
    return request_ownership(rows, stable._frame_index(rows))


def _recordings(monkeypatch):
    windows, bodies, sends = _recorded_k2_sends(monkeypatch)
    layouts = [stable.recorded_window_layout(result["messages"]) for result, _ in sends]
    return windows, bodies, sends, layouts


def test_chief_marker_rows_in_the_exchange_are_ordinary_content(monkeypatch):
    window = _window()
    echo = "[Current source: pkg/a.py] quoted by the model"
    probes = [
        {"role": "assistant", "content": echo, "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": echo},
        {"role": "user", "content": echo},
        {"role": "assistant", "content": window["content"]},
    ]
    body = _body([window])
    body["messages"][-1:-1] = probes
    assert len(stable._window_entries(body["messages"], _ownership(body["messages"]))) == 1
    runtime = SimpleNamespace(_stable_wire_state=None)
    result, _, journal = _invoke(monkeypatch, body, runtime)
    assert not _events(journal, "stable_wire_failed_open")
    projected = json.loads(json.dumps(result))["messages"]
    assert projected[2] == window
    status = stable._frame_index(projected) + 2
    assert projected[status + 1:] == [*probes, body["messages"][-1]]
    assert stable.canonicalize_stable_wire(projected)[0][-5:] == body["messages"][-5:]
    assert stable.recorded_window_contents(projected) == [window["content"]]
    assert len(runtime._stable_wire_state["delivered"]) == 1


@pytest.mark.parametrize("position", ["before_window", "after_window"])
def test_chief_malformed_marker_before_frame_is_ordinary_content(monkeypatch, position):
    window = _window()
    probe = {"role": "user", "content": "[Current source: pkg/b.py] not a window"}
    rows = [probe, window] if position == "before_window" else [window, probe]
    body = _body(rows)
    entries = stable._window_entries(body["messages"], _ownership(body["messages"]))
    assert [row["identity"]["path"] for row in entries] == ["pkg/a.py"]
    runtime = SimpleNamespace(_stable_wire_state=None)
    result, _, journal = _invoke(monkeypatch, body, runtime)
    assert not _events(journal, "stable_wire_failed_open")
    projected = json.loads(json.dumps(result))["messages"]
    assert projected[2] == window
    assert projected[stable._frame_index(projected) + 3] == probe
    canonical, _ = stable.canonicalize_stable_wire(projected)
    assert canonical[stable._frame_index(canonical) - 1] == probe
    assert stable._order_preserved(body["messages"], projected)
    assert stable.recorded_window_contents(projected) == [window["content"]]
    assert len(runtime._stable_wire_state["delivered"]) == 1


def test_chief_window_entries_ignore_window_shaped_rows_after_frame():
    window, second = _window(), _window(path="pkg/b.py")
    body = _body([window])
    body["messages"].append(second)
    assert [row["identity"]["path"] for row in stable._window_entries(
        body["messages"], _ownership(body["messages"]))] == ["pkg/a.py"]
    # Current helpers require explicit ownership, including window-only callers.
    windows = [window, second]
    owner = stable.RequestOwnership(0, 2, 3, 2, 2)
    assert len(stable._window_entries(windows, owner)) == 2
    assert stable._window_entries(windows) == []


@pytest.mark.parametrize("metadata", [{"name": "native-user"}, {"tool_call_id": "call-0"}])
@pytest.mark.parametrize("position", ["after_first_window", "before_frame"])
def test_chief_metadata_row_before_frame_keeps_its_position(monkeypatch, metadata, position):
    # Captured provenance now preserves these rows instead of guessing ownership
    # from their names and failing open. Existing named host-panel tests require it.
    body = _final_cache_shaped_body()
    rows = body["messages"]
    owner = _ownership(rows)
    first_window = next(i for i, row in enumerate(rows)
                        if stable._is_current_window(row, i, ownership=owner))
    planted = {"role": "user", "content": "[Host fact] planted named notice", **metadata}
    rows.insert(first_window + 1 if position == "after_first_window" else owner.frame_index, planted)
    original = copy.deepcopy(body)
    result, _, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert not _events(journal, "stable_wire_failed_open")
    assert body == original
    recorded = json.loads(json.dumps(result))["messages"]
    canonical, _ = stable.canonicalize_stable_wire(recorded)
    assert canonical.index(planted) < stable._frame_index(canonical)
    assert [row for row in canonical if row == planted] == [planted]
    assert stable._order_preserved(rows, recorded)


def _misfile_last_panel(projected):
    rows = projected["messages"]
    frame = stable._frame_index(rows)
    order = projection_order(rows, frame)
    changed = ProjectionOrder(order.prefix_end, order.tail_count - 1,
                              order.notices, order.newest_count)
    rows[frame + 2]["content"] = rows[frame + 2]["content"].removesuffix(order.encode()) + changed.encode()


@pytest.mark.parametrize("metadata", [{"name": "native-user"}, {"tool_call_id": "call-0"}])
def test_chief_order_proof_rejects_misfiled_pre_frame_row(monkeypatch, metadata):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    body = _final_cache_shaped_body()
    rows = body["messages"]
    planted = {"role": "user", "content": "[Host fact] planted named notice", **metadata}
    rows.insert(stable._frame_index(rows), planted)
    owner = _ownership(rows)
    current = stable._window_entries(rows, owner)
    projected, _, _ = stable._project_body(body, [], current, 1, ownership=owner)
    assert stable._order_preserved(rows, projected["messages"])
    _misfile_last_panel(projected)
    canonical, _ = stable.canonicalize_stable_wire(projected["messages"])
    assert canonical.index(planted) > stable._frame_index(canonical)
    assert not stable._order_preserved(rows, projected["messages"])

    real = stable._project_body

    def misfile(*args, **kwargs):
        result, status, frame = real(*args, **kwargs)
        _misfile_last_panel(result)
        return result, status, frame

    monkeypatch.setattr(stable, "_project_body", misfile)
    runtime = SimpleNamespace(_stable_wire_state=None)
    result, _, journal = _invoke(monkeypatch, body, runtime)
    failed = _events(journal, "stable_wire_failed_open")
    assert result is body
    assert [event["reason"] for event in failed] == ["projection_moves_frame"]
    assert "planted" not in json.dumps(failed)
    assert runtime._stable_wire_state is None


def test_chief_plain_newest_turn_before_frame_is_accepted(monkeypatch):
    body = _final_cache_shaped_body()
    rows = copy.deepcopy(body["messages"])
    result, _, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert not _events(journal, "stable_wire_failed_open")
    assert stable._order_preserved(rows, result["messages"])
    canonical, _ = stable.canonicalize_stable_wire(result["messages"])
    frame = stable._frame_index(canonical)
    assert canonical[frame - 1]["content"] == ADVISORY
    assert canonical[frame - 2]["content"] == SUMMARY


def test_chief_order_proof_sees_frame_cross_native_row(monkeypatch):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    body = _final_cache_shaped_body()
    rows = body["messages"]
    owner = _ownership(rows)
    projected, _, _ = stable._project_body(
        body, [], stable._window_entries(rows, owner), 1, ownership=owner)
    assert stable._order_preserved(rows, projected["messages"])
    swapped = copy.deepcopy(projected["messages"])
    frame = stable._frame_index(swapped)
    assistant = next(i for i, row in enumerate(swapped) if row.get("role") == "assistant")
    swapped.insert(frame, swapped.pop(assistant))
    assert not stable._order_preserved(rows, swapped)


def test_chief_replay_without_windows_fails_open_on_every_send(monkeypatch):
    _, bodies, _, _ = _recordings(monkeypatch)
    _replay_with(monkeypatch, lambda: None)
    runtime = SimpleNamespace(_stable_wire_state=None)
    for body in bodies:
        sent = copy.deepcopy(body)
        replayed, _, journal = _invoke(monkeypatch, sent, runtime,
                                       meter=ForcedRelayoutMeter(max_sources=99))
        assert replayed is sent and replayed == body
        assert [e["reason"] for e in _events(journal, "stable_wire_failed_open")] == [
            "replay_layout_mismatch"]
        assert not _events(journal, "stable_wire_epoch")
    assert runtime._stable_wire_state is None


def test_chief_recorded_layout_carries_epoch(monkeypatch):
    windows, _, _, layouts = _recordings(monkeypatch)
    first, second = layouts
    assert first[1] == 1 and second[1] == 2
    assert second[0] == [str(windows[n]["content"]) for n in "ABFE"]
    assert stable.recorded_window_layout([{"role": "user", "content": "plain"}]) is None


def test_chief_unparseable_status_has_no_recorded_layout(monkeypatch):
    _, _, recorded, _ = _recordings(monkeypatch)
    messages = copy.deepcopy(recorded[0][0]["messages"])
    frame = stable._frame_index(messages)
    status = messages[frame + 2]
    status["content"] = status["content"].replace("Epoch 1", "Epoch one")
    # Current API refuses the whole replay layout instead of returning epoch=None.
    assert stable.recorded_window_layout(messages) is None


def test_chief_identical_windows_at_another_epoch_fail_open(monkeypatch):
    _, bodies, _, layouts = _recordings(monkeypatch)
    _replay_with(monkeypatch, lambda: (layouts[0][0], layouts[0][1] + 1))
    runtime = SimpleNamespace(_stable_wire_state=None)
    body = copy.deepcopy(bodies[0])
    result, _, journal = _invoke(monkeypatch, body, runtime,
                                meter=ForcedRelayoutMeter(max_sources=99))
    assert result is body and result == bodies[0]
    assert [e["reason"] for e in _events(journal, "stable_wire_failed_open")] == [
        "replay_layout_mismatch"]
    mismatch, = _events(journal, "stable_wire_replay_mismatch")
    assert mismatch["epoch"] == 1 and mismatch["recorded_epoch"] == 2
    assert mismatch["recorded_window_count"] == mismatch["projected_window_count"] == 4
    assert mismatch["shared_prefix_windows"] == 4
    assert runtime._stable_wire_state is None


def test_chief_legacy_window_list_obeys_current_epoch_requirement(monkeypatch):
    _, bodies, _, layouts = _recordings(monkeypatch)
    _replay_with(monkeypatch, lambda: layouts[0][0])
    runtime = SimpleNamespace(_stable_wire_state=None)
    body = copy.deepcopy(bodies[0])
    result, _, journal = _invoke(monkeypatch, body, runtime)
    assert result is body
    assert [e["reason"] for e in _events(journal, "stable_wire_failed_open")] == [
        "recorded_row_ownership_unavailable"]
    assert runtime._stable_wire_state is None


@pytest.mark.parametrize("bad", [
    ["ok", 3], (["ok", 3], 1), (["ok"], "1"), (["ok"], True),
])
def test_chief_provider_rejects_malformed_recording(monkeypatch, bad):
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", None)
    candidate.install_replay_stable_wire_windows_provider(lambda: bad)
    with pytest.raises(ValueError):
        candidate.replay_stable_wire_windows()


def test_chief_unreproducible_recorded_order_fails_open(monkeypatch):
    _, bodies, _, layouts = _recordings(monkeypatch)
    rows, epoch = layouts[1]
    layouts[1] = ([rows[2], rows[3], rows[0], rows[1]], epoch)
    cursor = {"send": 0}
    _replay_with(monkeypatch, lambda: layouts[cursor["send"]])
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, copy.deepcopy(bodies[0]), runtime)
    previous = copy.deepcopy(runtime._stable_wire_state)
    cursor["send"] = 1
    body = copy.deepcopy(bodies[1])
    result, _, journal = _invoke(monkeypatch, body, runtime)
    assert result is body
    mismatch, = _events(journal, "stable_wire_replay_mismatch")
    assert mismatch["recorded_window_count"] == 4
    assert mismatch["shared_prefix_windows"] < 4
    assert _events(journal, "stable_wire_failed_open")[0]["reason"] == "replay_layout_mismatch"
    assert runtime._stable_wire_state == previous


@pytest.mark.parametrize("mismatch_after_trim", [1, 2])
def test_chief_replay_rechecks_window_counts_after_trims(monkeypatch, mismatch_after_trim):
    live_body = _layout_body(6)
    live_trim = UnderTrimmer(live_body, rows_per_reserve=0.05)
    live, live_capture = _render(
        monkeypatch, live_body, meter=SizeMeter(), inputs=_inputs(70), trim=live_trim)
    assert live_trim.calls == [(28, 256), (46, 512)]
    recorded = stable.recorded_window_layout(live["messages"])
    _replay_with(monkeypatch, lambda: recorded)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_RESERVES_PROVIDER",
                        lambda: live_capture._stable_wire_reserves_record)
    body = _layout_body(6)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)

    def trim(tokens, byte_count):
        trimmer(tokens, byte_count)
        if len(trimmer.calls) == mismatch_after_trim:
            window = next(row for row in body["messages"]
                          if str(row.get("content", "")).startswith("[Current source:"))
            window["content"] = window["content"].replace("value = 1", "value = 2")
        return True

    runtime = SimpleNamespace(_stable_wire_state=None)
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(70),
                              trim=trim, runtime=runtime)
    assert trimmer.calls == live_trim.calls
    mismatch, = _events(capture.journal, "stable_wire_replay_mismatch")
    assert _failed_open_events(capture)[0]["reason"] == "replay_layout_mismatch"
    assert mismatch["recorded_window_count"] == mismatch["projected_window_count"] == len(recorded[0])
    assert mismatch["shared_prefix_windows"] == 0
    assert not any("[Current source:" in str(value) for value in mismatch.values())
    assert not stable.is_stable_wire_projection(result["messages"])
    assert runtime._stable_wire_state is None


def test_chief_missing_recording_fails_open_before_trim(monkeypatch):
    _replay_with(monkeypatch, lambda: None)
    body = _layout_body(6)
    before = copy.deepcopy(body)
    trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
    result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(70), trim=trimmer)
    assert trimmer.calls == []
    assert body == before
    assert not stable.is_stable_wire_projection(result["messages"])
    assert capture._final_cache_layout_record["stable_wire"] == "failed_open"
    assert _failed_open_events(capture)[0]["reason"] == "replay_layout_mismatch"
    assert len(_events(capture.journal, "stable_wire_replay_mismatch")) == 1
