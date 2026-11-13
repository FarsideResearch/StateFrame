"""Host ownership comes from capture, never from a row's text or name."""
from __future__ import annotations

import copy
import json
import re
from types import SimpleNamespace

import pytest

import kernel_stable_wire as stable
from kernel_cache_layout_candidate import _checked_final_cache_projection
from kernel_request_source_window import source_delivery_satisfied
from test_newest_user_turn_survives_wire_projection import (
    ADVISORY, NOTICE_CONTENTS, _final_cache_shaped_body,
)
from test_stable_wire import Journal, Meter, _body, _final_layout_body, _invoke, _window


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_astra_native_frame_counterexample(monkeypatch, role):
    template = _final_layout_body()
    template["messages"][-1:-1] = copy.deepcopy(
        _final_cache_shaped_body()["messages"][-3:-1])
    native = next(row for row in template["messages"] if row.get("role") == role)
    native["content"] = "[StateFrame]\n{}"
    # Astra's body is valid for the plain final-layout projector.
    plain = _checked_final_cache_projection(template)
    result, capture, _ = _invoke(monkeypatch, plain, SimpleNamespace(_stable_wire_state=None))
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    recorded = json.loads(json.dumps(result))["messages"]
    frame = next(i for i, r in enumerate(plain["messages"]) if r.get("role") == "assistant") - 1
    assert stable.materialize_stable_frame(recorded) == plain["messages"][frame]["content"]
    assert stable.recorded_window_layout(recorded) == ([_window()["content"]], 1)
    assert stable._order_preserved(plain["messages"], recorded)
    assert stable.canonicalize_stable_wire(recorded)[0] == plain["messages"]
    from kernel_cache_layout_view import semantic_messages
    assert semantic_messages(recorded) == semantic_messages(plain["messages"])


@pytest.mark.parametrize("retain", ["on", "off"])
def test_astra_changed_system_preface_counterexample(monkeypatch, retain):
    runtime = SimpleNamespace(_stable_wire_state=None)
    body = _final_cache_shaped_body()
    first, _, _ = _invoke(monkeypatch, body, runtime, retain=retain)
    second_body = copy.deepcopy(body)
    second_body["messages"][0]["content"] += " changed"
    second, _, journal = _invoke(monkeypatch, second_body, runtime, retain=retain)
    assert first["messages"][0] != second["messages"][0]
    assert runtime._stable_wire_state["epoch"] == 2
    assert [e["reason"] for e in journal.events if e["kind"] == "stable_wire_epoch"] == ["structural"]


@pytest.mark.parametrize("retain", ["on", "off"])
@pytest.mark.parametrize("change", ["task", "system_metadata", "system_key_order", "extra_system"])
def test_every_serialized_preface_change_starts_an_epoch(monkeypatch, retain, change):
    runtime = SimpleNamespace(_stable_wire_state=None)
    body = _final_cache_shaped_body()
    _invoke(monkeypatch, body, runtime, retain=retain)
    changed = copy.deepcopy(body)
    if change == "task":
        changed["messages"][1]["content"] += "\n追加 instructions"
    elif change == "system_metadata":
        changed["messages"][0]["name"] = "host_system"
    elif change == "system_key_order":
        changed["messages"][0] = dict(reversed(list(changed["messages"][0].items())))
    else:
        changed["messages"].insert(1, {"role": "system", "content": "second system"})
    _invoke(monkeypatch, changed, runtime, retain=retain)
    assert runtime._stable_wire_state["epoch"] == 2


@pytest.mark.parametrize("retain", ["on", "off"])
def test_per_send_tail_changes_do_not_enter_preface_signature(monkeypatch, retain):
    runtime = SimpleNamespace(_stable_wire_state=None)
    body = _final_cache_shaped_body()
    first, _, _ = _invoke(monkeypatch, body, runtime, retain=retain)
    changed = copy.deepcopy(body)
    for row in changed["messages"]:
        if row.get("content") == ADVISORY or row.get("role") in {"assistant", "tool"}:
            row["content"] = "Task: per-send native or captured text"
    second, _, _ = _invoke(monkeypatch, changed, runtime, retain=retain)
    assert runtime._stable_wire_state["epoch"] == 1
    assert first["messages"][:3] == second["messages"][:3]


@pytest.mark.parametrize("marker", [
    "[StateFrame]\n", "[Current source:", "[Deferred required source paths]\n",
    stable.DIET_NOTICE_HEADER + "\n", "[Host fact]", "[Source windows status]", "[Source pool",
])
@pytest.mark.parametrize("role", ["assistant", "tool", "newest_user"])
def test_host_markers_in_unowned_rows_never_raise(monkeypatch, marker, role):
    body = _final_cache_shaped_body()
    # Match the semantic layout's source-before-panels order, so whole-row
    # equality below checks opaque rows without exempting host panel movement.
    body["messages"][2], body["messages"][3] = body["messages"][3], body["messages"][2]
    row = next(r for r in body["messages"] if (
        r.get("content") == ADVISORY if role == "newest_user" else r.get("role") == role))
    row["content"] = marker + "{}"
    capture = SimpleNamespace(journal=Journal(), request_id="marker-matrix",
                              _framefit5_newest_user_suffix=(copy.deepcopy(row),)
                              if role == "newest_user" else ())
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_a, **_k: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _r: True)
    runtime = SimpleNamespace(_stable_wire_state=None)
    result = stable.project_stable_wire(
        capture, runtime, body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    disposition = capture._final_cache_layout_record["stable_wire"]
    assert disposition in {"applied", "failed_open"}
    if disposition == "failed_open":
        assert result == body
        assert runtime._stable_wire_state is None
    else:
        recorded = json.loads(json.dumps(result))["messages"]
        assert stable.canonicalize_stable_wire(recorded)[0] == body["messages"]
        canonical = copy.deepcopy(stable.canonicalize_stable_wire(recorded)[0])
        assert canonical[stable._frame_index(canonical)]["content"] != marker + "{}"
        assert stable.recorded_window_contents(recorded) == [_window()["content"]]
        assert stable._order_preserved(body["messages"], recorded)
        assert row in recorded


@pytest.mark.parametrize("where", ["captured_multirow", "native_user", "final_user"])
def test_frame_shaped_user_text_is_opaque_through_every_readback(monkeypatch, where):
    body = _final_cache_shaped_body()
    rows = body["messages"]
    row = {"role": "user", "content": "[StateFrame]\n{}"}
    captured = ()
    if where == "captured_multirow":
        at = next(i for i, r in enumerate(rows) if r.get("content") == ADVISORY)
        rows[at:at + 1] = [row, {"role": "user", "content": "second captured row"}]
        captured = copy.deepcopy(rows[at:at + 2])
    elif where == "native_user":
        rows.insert(-1, row)
    else:
        rows[-1] = row
    _invoke(monkeypatch, _final_cache_shaped_body(), SimpleNamespace(_stable_wire_state=None))
    capture = SimpleNamespace(journal=Journal(), request_id="opaque-frame-user",
                              _framefit5_newest_user_suffix=captured)
    result = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    recorded = json.loads(json.dumps(result))["messages"]
    assert stable.recorded_window_layout(recorded) == ([_window()["content"]], 1)
    assert stable._order_preserved(rows, recorded)
    assert row in stable.canonicalize_stable_wire(recorded)[0]
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    from kernel_cache_layout_view import semantic_messages
    canonical = semantic_messages(recorded)
    assert row in canonical
    assert semantic_messages(copy.deepcopy(canonical)) == canonical


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_native_frame_counterexample_through_real_final_renderer(monkeypatch, role):
    from kernel_cache_layout_candidate import render_final_cache_candidate
    from kernel_cache_layout_view import (
        inspect_layout, semantic_index_map, semantic_messages, source_messages, volatile_blocks,
    )

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    template = _final_layout_body()
    template["messages"][-1:-1] = copy.deepcopy(
        _final_cache_shaped_body()["messages"][-3:-1])
    native = next(row for row in template["messages"] if row.get("role") == role)
    native["content"] = "[StateFrame]\n{}"
    capture = SimpleNamespace(journal=Journal(), request_id="real-native-frame",
                              _framefit5_newest_user_suffix=())
    result = render_final_cache_candidate(
        capture, SimpleNamespace(_stable_wire_state=None), template, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {"new_output_target": True, "workspace_authorization": {
                    "scope": "test-workspace", "generation": "1"}}, "sources": {}})
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    assert source_delivery_satisfied(capture.request_source_final_delivery)
    recorded = json.loads(json.dumps(result))["messages"]
    assert native in semantic_messages(recorded)
    index, payload, pages, end = inspect_layout(recorded)
    assert index == end == stable._frame_index(recorded)
    assert payload == json.loads(template["messages"][2]["content"].split("\n", 1)[1])
    assert pages == frozenset()
    assert semantic_index_map(recorded)[index] == 2
    assert source_messages(recorded) == recorded
    assert isinstance(volatile_blocks(recorded), tuple)


@pytest.mark.parametrize("error_type", [stable.StableWireError, "cache_layout"])
def test_unproven_projection_fails_open_without_publishing_epoch(monkeypatch, error_type):
    from kernel_cache_layout_view import CacheLayoutError

    body = _final_cache_shaped_body()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, body, runtime)
    previous = copy.deepcopy(runtime._stable_wire_state)

    def rejected(*_args, **_kwargs):
        error = CacheLayoutError if error_type == "cache_layout" else error_type
        raise error("model-controlled contents failed proof")

    monkeypatch.setattr(stable, "_project_body", rejected)
    result, capture, _ = _invoke(monkeypatch, body, runtime)
    assert result is body
    assert capture._final_cache_layout_record["stable_wire_reason"] == "unproven_stable_wire"
    assert runtime._stable_wire_state == previous


@pytest.mark.parametrize("retain", ["on", "off"])
@pytest.mark.parametrize("named", [False, True], ids=["unnamed", "host_notice"])
def test_changing_host_notice_before_windows_preserves_prefix_or_bumps_epoch(
        monkeypatch, retain, named):
    runtime = SimpleNamespace(_stable_wire_state=None)
    sends = []
    for number in (1, 2):
        body = _final_cache_shaped_body()
        row = {"role": "user", "content": f"[Host fact] send {number}"}
        if named:
            row["name"] = "host_notice"
        body["messages"].insert(2, row)
        result, capture, _ = _invoke(monkeypatch, body, runtime, retain=retain)
        assert capture._final_cache_layout_record["stable_wire"] == "applied"
        rows = result["messages"]
        frame = stable._frame_index(rows)
        status = next(r["content"] for r in rows if r.get("name") == stable.STATUS_NAME)
        epoch = int(re.search(r"(?m)^Epoch (\d+)$", status).group(1))
        sends.append((epoch, json.dumps(rows[:frame], ensure_ascii=False).encode()))
        assert rows.index(row) > frame
    assert sends[0][0] != sends[1][0] or sends[0][1] == sends[1][1]


def test_named_host_after_unnamed_host_survives_real_source_verification(monkeypatch):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    template = _final_layout_body()
    template["messages"][2:2] = [
        {"role": "user", "content": "[Host fact] unnamed"},
        {"role": "user", "name": "host_notice", "content": "[Host fact] send 1"},
    ]
    body = _checked_final_cache_projection(template)
    capture = SimpleNamespace(journal=Journal(), request_id="named-host",
                              _framefit5_newest_user_suffix=())
    result = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {"new_output_target": True, "workspace_authorization": {
                    "scope": "test-workspace", "generation": "1"}}, "sources": {}})
    assert stable.is_stable_wire_projection(result["messages"])
    assert source_delivery_satisfied(capture.request_source_final_delivery)
    # Provenance must survive recording, with no extra wire-message keys.
    recorded = json.loads(json.dumps(result))["messages"]
    canonical, _ = stable.canonicalize_stable_wire(recorded)
    assert stable._order_preserved(body["messages"], recorded)
    assert canonical[-1] == body["messages"][-1]
    assert [r for r in canonical if r.get("name") == "host_notice"] == [template["messages"][3]]


@pytest.mark.parametrize("name", [stable.STATUS_NAME, stable.VOLATILE_NAME])
def test_host_name_cannot_impersonate_projection_metadata_in_the_final_renderer(monkeypatch, name):
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    template = _final_layout_body()
    host = {"role": "user", "name": name, "content": "[Host fact] send 1"}
    template["messages"].insert(2, host)
    capture = SimpleNamespace(journal=Journal(), request_id="reserved-host-name",
                              _framefit5_newest_user_suffix=())
    result = render_final_cache_candidate(
        capture, SimpleNamespace(_stable_wire_state=None), template, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {"new_output_target": True, "workspace_authorization": {
                    "scope": "test-workspace", "generation": "1"}}, "sources": {}})
    rows = result["messages"]
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    assert source_delivery_satisfied(capture.request_source_final_delivery)
    assert rows.index(host) > stable._frame_index(rows)


def test_off_retains_the_legacy_plain_fallback_for_a_reserved_host_name(monkeypatch):
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    template = _final_layout_body()
    template["messages"].insert(2, {
        "role": "user", "name": stable.VOLATILE_NAME, "content": "[Host fact] send 1"})
    capture = SimpleNamespace(journal=Journal(), request_id="reserved-name-off")
    result = render_final_cache_candidate(
        capture, SimpleNamespace(), template, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {"new_output_target": True, "workspace_authorization": {
                    "scope": "test-workspace", "generation": "1"}}, "sources": {}})
    assert json.dumps(result, ensure_ascii=False) == json.dumps(template, ensure_ascii=False)


def test_notice_shaped_captured_newest_user_is_never_moved_or_exempt(monkeypatch):
    text = "[Deferred required source paths]\nThis is my newest user turn"
    body = _final_cache_shaped_body()
    row = next(r for r in body["messages"] if r.get("content") == ADVISORY)
    row["content"] = text
    original = copy.deepcopy(body["messages"])
    capture = SimpleNamespace(journal=Journal(), request_id="captured-notice",
                              _framefit5_newest_user_suffix=(copy.deepcopy(row),))
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_a, **_k: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _r: True)
    result = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    canonical, _ = stable.canonicalize_stable_wire(result["messages"])
    assert stable._order_preserved(original, result["messages"])
    assert canonical[5] == row
    assert canonical[6:] == original[6:]
    broken = copy.deepcopy(result["messages"])
    at = broken.index(row)
    broken[at - 1], broken[at] = broken[at], broken[at - 1]
    assert not stable._order_preserved(original, broken)


@pytest.mark.parametrize("content", [*NOTICE_CONTENTS, _window()["content"]])
@pytest.mark.parametrize("name", [None, "host_notice", stable.STATUS_NAME, stable.VOLATILE_NAME])
def test_every_notice_shape_in_a_captured_user_row_stays_opaque(monkeypatch, content, name):
    body = _final_cache_shaped_body()
    row = next(r for r in body["messages"] if r.get("content") == ADVISORY)
    row["content"] = content
    if name is not None:
        row["name"] = name
    capture = SimpleNamespace(journal=Journal(), request_id="opaque-user",
                              _framefit5_newest_user_suffix=(copy.deepcopy(row),))
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_a, **_k: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _r: True)
    result = stable.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    recorded = json.loads(json.dumps(result))["messages"]
    canonical, _ = stable.canonicalize_stable_wire(recorded)
    assert canonical[5] == row
    exchange = next(i for i, r in enumerate(recorded) if r.get("role") == "assistant")
    assert recorded[exchange - 1] == row
    assert stable.recorded_window_contents(recorded) == [_window()["content"]]
    assert stable._order_preserved(body["messages"], recorded)
    # Inspecting a recorded stable send does not depend on today's switch value.
    from kernel_cache_layout_view import semantic_messages
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    semantic = semantic_messages(recorded)
    assert semantic[5] == row
    assert semantic_messages(copy.deepcopy(semantic)) == semantic


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_source_shaped_native_exchange_content_is_never_a_host_window(monkeypatch, role):
    body = _final_cache_shaped_body()
    native = next(row for row in body["messages"] if row.get("role") == role)
    native["content"] = _window(path="native-text.py")["content"]
    result, _, _ = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    rows = json.loads(json.dumps(result))["messages"]
    assert stable.recorded_window_contents(rows) == [_window()["content"]]
    assert stable.canonicalize_stable_wire(rows)[0][-3:-1] == body["messages"][-3:-1]


def test_recorded_order_rejects_missing_or_malformed_ownership(monkeypatch):
    from kernel_stable_wire_ownership import ORDER_MARKER

    body = _final_cache_shaped_body()
    result, _, _ = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    for value in (None, {}, [2, 3, [3], 1], [2, 3, [True], 1], [2, 10000, [], 0]):
        rows = copy.deepcopy(result["messages"])
        status = rows[stable._frame_index(rows) + 2]
        status["content"] = status["content"].split(ORDER_MARKER, 1)[0]
        if value is not None:
            status["content"] += ORDER_MARKER + json.dumps(value)
        with pytest.raises(stable.StableWireError, match="row ownership"):
            stable.canonicalize_stable_wire(rows)


def test_final_renderer_carries_a_multirow_captured_turn_through_semantic_readback(monkeypatch):
    from kernel_cache_layout_candidate import render_final_cache_candidate
    from kernel_cache_layout_view import semantic_messages

    newest = [
        {"role": "user", "content": "[Deferred required source paths]\nThis is my newest user turn"},
        {"role": "user", "name": "host_notice", "content": "[Request diet: dropped source ranges]\nuser text"},
    ]
    template = _final_layout_body()
    template["messages"][-1:-1] = [
        *copy.deepcopy(newest), *copy.deepcopy(_final_cache_shaped_body()["messages"][-3:-1])]
    capture = SimpleNamespace(journal=Journal(), request_id="renderer-capture",
                              _framefit5_newest_user_suffix=copy.deepcopy(newest))
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_a, **_k: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _r: True)
    result = render_final_cache_candidate(
        capture, SimpleNamespace(_stable_wire_state=None), template, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}})
    recorded = json.loads(json.dumps(result))["messages"]
    assert capture._final_cache_layout_record["stable_wire"] == "applied"
    at = next(i for i, row in enumerate(recorded) if row.get("role") == "assistant")
    assert recorded[at - len(newest):at] == newest
    assert semantic_messages(recorded) == template["messages"]
    assert all(set(row) <= {"role", "content", "name", "tool_calls", "tool_call_id"}
               for row in recorded)


@pytest.mark.parametrize("mutation", ["edit", "insert", "move"])
def test_a_changed_captured_suffix_requests_replanning_before_projection(monkeypatch, mutation):
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded

    body = _final_cache_shaped_body()
    rows = body["messages"]
    at = next(i for i, row in enumerate(rows) if row.get("content") == ADVISORY)
    capture = SimpleNamespace(_framefit5_newest_user_suffix=(copy.deepcopy(rows[at]),))
    if mutation == "edit":
        rows[at]["content"] += " changed"
    elif mutation == "insert":
        rows.insert(at + 1, {"role": "user", "content": "[Host fact] late"})
    else:
        rows.insert(2, rows.pop(at))
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    runtime = SimpleNamespace(_stable_wire_state=None)
    with pytest.raises(FrameAllocationReplanNeeded, match="newest user suffix changed"):
        stable.project_stable_wire(capture, runtime, body, meter=Meter(), inputs={})
    assert runtime._stable_wire_state is None


def test_a_checkpoint_from_the_heuristic_layout_starts_a_new_epoch(monkeypatch):
    runtime = SimpleNamespace(_stable_wire_state=None)
    inputs = {"token_ceiling": 10_000, "byte_ceiling": 1_000_000, "source_plan": {},
              "sources": {}, "receipt": None, "switch_set": {}, "profile": ""}
    _invoke(monkeypatch, _body([_window()]), runtime, inputs=copy.deepcopy(inputs))
    # Recorded from 4a756535 for this exact fixture and explicit structural inputs.
    # Its prefix could contain a named host row, so none of its prefix is reusable
    # until the new ownership layout has established a fresh epoch.
    runtime._stable_wire_state["structure"] = (
        "6ebfef93e5c94a0bf9b6d0a62646eb5992e1ec55816953315b8a82d7033d6dec")
    _, _, journal = _invoke(monkeypatch, _body([_window()]), runtime, inputs=copy.deepcopy(inputs))
    assert runtime._stable_wire_state["epoch"] == 2
    assert [e["reason"] for e in journal.events if e["kind"] == "stable_wire_epoch"] == ["structural"]
