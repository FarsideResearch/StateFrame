"""Source ordering survives a plain-layout replay without moving user input."""

from __future__ import annotations

import json
from types import SimpleNamespace

import kernel_cache_layout_candidate as layout
from kernel_request_source_window import MODEL_WINDOW_SCHEMA


def _window(name: str) -> dict[str, str]:
    path = f"pkg/{name}.py"
    metadata = {
        "schema": MODEL_WINDOW_SCHEMA,
        "path": path,
        "owner": "recent_editable_read",
        "revision": f"revision-{name}",
        "ranges": [{"start_line": 1, "end_line": 1}],
        "total_lines": 1,
        "status": "complete",
    }
    return {"role": "user", "content": "\n".join((
        f"[Current source: {path}]", "[Model source window]",
        json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        "[Source window content]", "--- source lines 1-1 ---",
        f"VALUE = '{name}'", "[End source window]",
    ))}


def _bytes(body: object) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def test_recorded_plain_fallback_replay_restores_order_and_next_state(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setenv("STATEFRAME_PAGE_LOG", "off")
    original = SimpleNamespace()
    initial = {"messages": [_window("a"), _window("b")]}
    layout._ordered_failed_open_body(initial, original, stable_wire_enabled=False)
    resumed = SimpleNamespace(**vars(original))
    reversed_body = {"messages": [_window("b"), _window("a")]}

    expected = layout._ordered_failed_open_body(
        reversed_body, original, stable_wire_enabled=False)
    monkeypatch.setattr(layout, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "plain")
    actual = layout.render_final_cache_candidate(
        None, resumed, reversed_body, meter=None, inputs=None)
    assert _bytes(actual) == _bytes(expected)
    assert getattr(resumed, layout.FINAL_SOURCE_ORDER_STATE_ATTR) == \
        getattr(original, layout.FINAL_SOURCE_ORDER_STATE_ATTR)

    empty = {"messages": [{"role": "user", "content": "no source"}]}
    assert layout.render_final_cache_candidate(
        None, resumed, empty, meter=None, inputs=None) == empty
    assert getattr(resumed, layout.FINAL_SOURCE_ORDER_STATE_ATTR)["order"] == []


def test_source_block_after_state_frame_is_reordered():
    runtime = SimpleNamespace()
    layout._ordered_failed_open_body(
        {"messages": [_window("a"), _window("b")]},
        runtime, stable_wire_enabled=False)
    body = {"messages": [
        {"role": "user", "content": "[StateFrame]\n{}"},
        _window("b"),
        _window("a"),
    ]}
    ordered, state = layout._order_final_source_windows(body, runtime)
    assert ordered["messages"][1:] == [_window("a"), _window("b")]
    assert state is not None and len(state["order"]) == 2


def test_replay_fitter_retains_live_ordering_switch(monkeypatch):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.setattr(layout, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "plain")
    assert layout._final_source_order_enabled_for_fitting()
