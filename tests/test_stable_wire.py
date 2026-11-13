"""Unit coverage for the opt-in content-stable request projection."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from kernel_cache_layout_view import semantic_messages
from kernel_request_source_window import (
    SCHEMA as SOURCE_SCHEMA,
    WINDOW_CONTENT_MARKER,
    WINDOW_END_MARKER,
    WINDOW_MARKER,
    project_source_window,
)
from kernel_stable_wire import materialize_stable_frame, project_stable_wire


class Meter:
    """Deterministic stand-in for the request's token meter."""

    def measure(self, body):
        messages = body.get("messages", []) if isinstance(body, dict) else []
        if messages and all(str(row.get("content", "")).startswith("[Current source:")
                            for row in messages):
            size = sum(len(str(row.get("content", "")).split(
                WINDOW_CONTENT_MARKER + "\n", 1)[-1].split(WINDOW_END_MARKER, 1)[0].encode("utf-8"))
                       for row in messages)
            return size // 100, "tokenize"
        else:
            size = len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        return max(1, (size + 99) // 100), "tokenize"


class SourceMeter:
    """Count source messages independently while keeping the rest of a body cheap."""

    def __init__(self, costs, base_tokens=100):
        self.costs = costs
        self.base_tokens = base_tokens

    def measure(self, body):
        messages = body.get("messages", []) if isinstance(body, dict) else []
        sources = [row for row in messages
                   if str(row.get("content", "")).startswith("[Current source:")]
        source_tokens = sum(self.costs[row["content"]] for row in sources)
        if sources and len(sources) == len(messages):
            return source_tokens, "tokenize"
        return self.base_tokens + source_tokens, "tokenize"


class EstimatedMeter:
    """Use the 040b estimator method for request and source-only counts."""

    def __init__(self, source_method="estimate"):
        self.source_method = source_method
        self.source_measurements = []
        self.stable_measurements = []

    def measure(self, body):
        from kernel_context import token_estimate
        from kernel_stable_wire import is_stable_wire_projection

        messages = body.get("messages", []) if isinstance(body, dict) else []
        tokens = token_estimate(body)
        source_rows = [row for row in messages
                       if str(row.get("content", "")).startswith("[Current source:")]
        if messages and len(source_rows) == len(messages):
            method = self.source_method
            measured = 0 if method == "unavailable" else tokens
            self.source_measurements.append((
                tuple(row["content"] for row in source_rows), measured, method))
            return measured, method
        if is_stable_wire_projection(messages):
            self.stable_measurements.append((tokens, len(json.dumps(body).encode("utf-8"))))
        return tokens, "estimate"


class Journal:
    run_id = "stable-wire-test"

    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


def _revision(char: str) -> str:
    return "sha256:" + char * 64


def _window(path="pkg/a.py", revision=None, start=1, end=1, text="value = 1\n"):
    revision = revision or _revision("1")
    ranges = [{"start_line": start, "end_line": end, "anchors": [],
               "trailing_newline": text.endswith("\n")}]
    metadata = {
        "schema": SOURCE_SCHEMA,
        "path": path,
        "revision": revision,
        "sha256": revision.removeprefix("sha256:"),
        "total_lines": max(end, 1),
        "ranges": ranges,
        "omitted_lines": False,
        "continuation_receipt": {"status": "COMPLETE", "omitted_lines": 0,
                                 "expansion_handle": None},
        "page": {"tool": "read", "path": path, "revision": revision,
                 "max_lines": 1000, "max_chars": 64000, "full_revision_required": True},
    }
    source_lines = text if text.endswith("\n") else text + "\n"
    content = (
        f"[Current source: {path}]\n{WINDOW_MARKER}\n"
        + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + f"\n{WINDOW_CONTENT_MARKER}\n--- source lines {start}-{end} ---\n"
        + source_lines + WINDOW_END_MARKER
    )
    return {"role": "user", "content": content}


def _payload(*, rows=None, action="read-1", headcarry=None, recorded_events=None,
             open_sites=None, code_bookmarks=None):
    framefit5 = {
        "action_ledger": {"history_digest": {"count": 1, "digest": "h1"},
                          "last_test": {"result": "passed", "id": action},
                          "reads": [{"path": "pkg/a.py", "action": action}]},
        "headcarry": headcarry or {"heads": ["pkg/a.py"]},
        "rows": rows if rows is not None else [{"path": "pkg/a.py", "value": 1}],
        "stable_fact": "unchanged",
    }
    if open_sites is not None:
        framefit5["open_sites"] = open_sites
    if code_bookmarks is not None:
        framefit5["code_bookmarks"] = code_bookmarks
    state = {"framefit5": framefit5}
    result = {"state": state}
    if recorded_events is not None:
        result["recorded_events"] = recorded_events
    return result


def _body(windows=(), *, payload=None):
    payload = payload or _payload(recorded_events=[{"id": "event-1"}])
    frame = "[StateFrame]\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: keep source bytes stable"},
            *copy.deepcopy(list(windows)),
            {"role": "user", "content": frame},
            {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
        ],
        "tools": [],
    }


def _final_layout_body():
    payload = {"state": {"framefit5": {
        "owner_protocol": {
            "schema": "stateframe-framefit5-owner/1",
            "continuation_limits": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2",
        },
        "next_site": {"path": "pkg/a.py"},
        "action_ledger": {"reads": [], "edits": []},
    }}}
    frame = "[StateFrame]\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: stable wire parity"},
        {"role": "user", "content": frame},
        _window(),
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ], "tools": []}


def _invoke(monkeypatch, body, runtime, *, headroom=16_000, meter=None, inputs=None,
            retain="on"):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_RETAIN_PREFIX", retain)
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", str(headroom))
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda _plan, _messages, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    journal = Journal()
    capture = SimpleNamespace(journal=journal, request_id="request", step=1,
                              request_source_final_delivery=None)
    result = project_stable_wire(
        capture, runtime, body, meter=meter or Meter(),
        inputs=inputs or {"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                          "source_plan": {}, "sources": {}, "receipt": None},
    )
    return result, capture, journal


def _messages_with_prefix(rows, prefix):
    return [row for row in rows
            if isinstance(row.get("content"), str) and row["content"].startswith(prefix)]


def _status_rows(rows):
    return _messages_with_prefix(rows, "[Source windows status]")


def _serialized_stable_messages(messages, rows=None):
    """Exact serialized bytes of the stable head: up to the volatile row, or the first ``rows``."""
    stop = rows if rows is not None else next(
        index for index, row in enumerate(messages)
        if isinstance(row, dict) and row.get("name") == "stateframe_stable_wire_volatile"
    )
    return b"\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for row in messages[:stop]
    )


def _assert_stable_message_byte_prefix(first, second, rows=None):
    first_bytes = _serialized_stable_messages(first["messages"], rows)
    second_bytes = _serialized_stable_messages(second["messages"], rows)
    assert second_bytes.startswith(first_bytes)


def _estimated_window_sequence(monkeypatch, *, source_method, extra_sets=()):
    windows = [
        _window(path=f"pkg/{name}.py", revision=_revision(str(index)),
                text="x" * 30_000 + "\n")
        for index, name in enumerate(("a", "b", "c", "d"), start=1)
    ]
    runtime = SimpleNamespace(_stable_wire_state=None)
    meter = EstimatedMeter(source_method=source_method)
    inputs = {"token_ceiling": 33_344, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}, "receipt": None}
    current_sets = ([windows[0], windows[1]], [windows[1]], [windows[2]], [windows[3]],
                    *extra_sets)
    results = []
    events = []
    for current in current_sets:
        current = [windows[row] if isinstance(row, int) else row for row in current]
        result, _, journal = _invoke(
            monkeypatch, _body(current), runtime, headroom=16_000,
            meter=meter, inputs=copy.deepcopy(inputs))
        results.append(result)
        events.append(journal.events)
    return windows, results, events, meter


def test_windows_remain_byte_immutable_and_new_versions_append(monkeypatch):
    first = _window(revision=_revision("1"), text="value = 1\n")
    old_runtime = SimpleNamespace(_stable_wire_state=None)
    first_body = _body([first])
    first_result, _, _ = _invoke(monkeypatch, first_body, old_runtime)
    first_delivered = _messages_with_prefix(first_result["messages"], "[Current source:")[0]

    current_v2 = _window(revision=_revision("2"), text="value = 2\n")
    second_runtime = SimpleNamespace(_stable_wire_state=copy.deepcopy(old_runtime._stable_wire_state))
    second_body = _body([current_v2])
    second_result, _, _ = _invoke(monkeypatch, second_body, second_runtime)
    delivered = _messages_with_prefix(second_result["messages"], "[Current source:")
    assert delivered[0] == first_delivered
    assert delivered[1] == current_v2
    assert len(_status_rows(second_result["messages"])) == 1
    status = _status_rows(second_result["messages"])[0]["content"]
    assert "Not current, do not edit from these:" in status
    assert "pkg/a.py" in status and "lines 1 to 1" in status
    assert "pkg/a.py: current" in status
    assert "\u2014" not in status and "\u2013" not in status

    unrelated = _window(path="pkg/b.py", revision=_revision("3"), text="value = 3\n")
    third_result, _, _ = _invoke(monkeypatch, _body([unrelated]), second_runtime)
    third_sources = _messages_with_prefix(third_result["messages"], "[Current source:")
    assert third_sources == [first_delivered, current_v2, unrelated]
    third_status = _status_rows(third_result["messages"])[0]["content"]
    assert "Not current, do not edit from these:" in third_status
    assert "pkg/a.py" in third_status and "lines 1 to 1" in third_status

    # Planted control: same path/revision/range with one changed byte. Body 8: this used
    # to refuse and end the run; it is now a rerender relayout that never resends the
    # old bytes, so the delivered-bytes immutability holds through a fresh epoch instead.
    changed = copy.deepcopy(first)
    changed["content"] = changed["content"].replace("value = 1", "value = 9")
    control_runtime = SimpleNamespace(_stable_wire_state=copy.deepcopy(old_runtime._stable_wire_state))
    control_result, _, journal = _invoke(monkeypatch, _body([changed]), control_runtime)
    assert _messages_with_prefix(control_result["messages"], "[Current source:") == [changed]
    assert first_delivered not in control_result["messages"]
    _only_relayout(journal, "rerender")


def test_status_omits_not_current_section_when_all_delivered_windows_are_current(monkeypatch):
    result, _, _ = _invoke(monkeypatch, _body([_window()]),
                            SimpleNamespace(_stable_wire_state=None))
    status = _status_rows(result["messages"])[0]["content"]
    assert "[Source windows status]" in status
    assert "pkg/a.py: current" in status
    assert "Not current" not in status
    assert "\u2014" not in status and "\u2013" not in status


def test_model_source_windows_are_kept_byte_immutable(monkeypatch):
    old = project_source_window(_window(revision=_revision("1"))["content"])
    new = project_source_window(_window(revision=_revision("2"), text="value = 2\n")["content"])
    assert isinstance(old, str) and isinstance(new, str)
    runtime = SimpleNamespace(_stable_wire_state=None)
    first, _, _ = _invoke(monkeypatch, _body([{"role": "user", "content": old}]), runtime)
    second, _, _ = _invoke(monkeypatch, _body([{"role": "user", "content": new}]), runtime)
    assert _messages_with_prefix(first["messages"], "[Current source:") == [
        {"role": "user", "content": old},
    ]
    delivered = _messages_with_prefix(second["messages"], "[Current source:")
    assert delivered == [{"role": "user", "content": old}, {"role": "user", "content": new}]
    assert _revision("1") in _status_rows(second["messages"])[0]["content"]


def test_frame_split_moves_whole_sections_and_rejects_volatile_key_in_stable_part(monkeypatch):
    payload = _payload(
        recorded_events=[{"id": "event-1"}],
        open_sites=[{"path": "pkg/open.py", "reason": "unfinished"}],
        code_bookmarks=[{"path": "pkg/a.py", "line": 8}],
    )
    original = _body([_window()], payload=payload)
    original_frame = next(row["content"] for row in original["messages"]
                          if row.get("content", "").startswith("[StateFrame]\n"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    projected, _, _ = _invoke(monkeypatch, original, runtime)
    assert materialize_stable_frame(projected["messages"]) == original_frame
    assert semantic_messages(projected["messages"])[0]["content"] == "system"

    stable = json.loads(next(row["content"] for row in projected["messages"]
                             if row.get("content", "").startswith("[StateFrame]\n")).split("\n", 1)[1])
    for path in ("action_ledger", "headcarry", "open_sites", "code_bookmarks"):
        assert path not in stable["state"]["framefit5"]
    assert "recorded_events" not in stable

    volatile_row = next(row for row in projected["messages"]
                        if row.get("name") == "stateframe_stable_wire_volatile")
    volatile = json.loads(volatile_row["content"].split("\n", 1)[1])
    expected_payload = json.loads(original_frame.split("\n", 1)[1])
    assert set(volatile) == {"schema", "data"}
    assert "fields" not in volatile
    data = volatile["data"]
    for key in ("action_ledger", "headcarry", "open_sites", "code_bookmarks"):
        assert data["state"]["framefit5"][key] == expected_payload["state"]["framefit5"][key]
    assert data["recorded_events"] == expected_payload["recorded_events"]
    assert "input_orders" not in volatile

    # A stable half that still carries a volatile framefit5 key is ambiguous.
    broken = copy.deepcopy(projected["messages"])
    frame_row = next(row for row in broken if row.get("content", "").startswith("[StateFrame]\n"))
    frame_payload = json.loads(frame_row["content"].split("\n", 1)[1])
    frame_payload["state"]["framefit5"]["action_ledger"] = {"history_digest": {"count": 1}}
    frame_row["content"] = "[StateFrame]\n" + json.dumps(
        frame_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="volatile"):
        materialize_stable_frame(broken)


def test_moved_host_and_guidance_rows_keep_recorded_order_across_frame(monkeypatch):
    body = _body([_window()])
    messages = body["messages"]
    frame_index = next(index for index, row in enumerate(messages)
                       if str(row.get("content", "")).startswith("[StateFrame]\n"))
    before = {"role": "user", "content": "[Host fact] before-frame"}
    coverage = {"role": "user", "content": "kept 1 selected source windows; about 322 chars of source fit"}
    guidance = {"role": "user", "content": "[tool guidance] middle"}
    after = {"role": "user", "content": "[Host fact] after-frame"}
    messages[2:2] = [before, coverage]
    frame_index += 1
    messages[frame_index + 1:frame_index + 1] = [guidance, after]

    result, _, _ = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    moved = [row["content"] for row in result["messages"]
             if str(row.get("content", "")).startswith(
                 ("[Host fact]", "[tool guidance]", "kept "))]
    assert moved == [before["content"], coverage["content"], guidance["content"], after["content"]]
    positions = {prefix: next(i for i, row in enumerate(result["messages"])
                              if str(row.get("content", "")).startswith(prefix))
                 for prefix in ("[StateFrame]\n", "[StateFrame volatile]", "[Source windows status]")}
    assert positions["[StateFrame]\n"] + 1 == positions["[StateFrame volatile]"]
    assert positions["[StateFrame volatile]"] + 1 == positions["[Source windows status]"]
    first_moved = min(i for i, row in enumerate(result["messages"])
                      if row.get("content") in moved)
    assert positions["[Source windows status]"] < first_moved
    assert [row.get("content") for row in result["messages"][first_moved:first_moved + len(moved)]] == moved


def test_stable_lists_keep_current_order_and_values_without_overlays(monkeypatch):
    window = _window()
    runtime = SimpleNamespace(_stable_wire_state=None)
    first = _body([window], payload=_payload(rows=[
        {"id": "first", "value": 1}, {"id": "second", "value": 2},
    ]))
    _invoke(monkeypatch, first, runtime)
    reordered = _body([window], payload=_payload(rows=[
        {"id": "second", "value": 3}, {"id": "first", "value": 4},
    ]))
    projected, _, _ = _invoke(monkeypatch, reordered, runtime)
    stable_content = next(row["content"] for row in projected["messages"]
                          if row.get("content", "").startswith("[StateFrame]\n"))
    stable = json.loads(stable_content.split("\n", 1)[1])
    stable_rows = stable["state"]["framefit5"]["rows"]
    assert [row["id"] for row in stable_rows] == ["second", "first"]
    assert [row["value"] for row in stable_rows] == [3, 4]
    original_frame = next(row["content"] for row in reordered["messages"]
                          if row.get("content", "").startswith("[StateFrame]\n"))
    assert materialize_stable_frame(projected["messages"]) == original_frame
    rebuilt = json.loads(materialize_stable_frame(projected["messages"]).split("\n", 1)[1])
    rows = rebuilt["state"]["framefit5"]["rows"]
    assert [row["id"] for row in rows] == ["second", "first"]
    assert [row["value"] for row in rows] == [3, 4]


def test_nested_stable_lists_keep_current_order(monkeypatch):
    runtime = SimpleNamespace(_stable_wire_state=None)
    first = _body([_window()], payload=_payload(rows=[
        {"id": "a", "children": [{"id": "x"}, {"id": "y"}]},
        {"id": "b", "children": [{"id": "p"}, {"id": "q"}]},
    ]))
    _invoke(monkeypatch, first, runtime)
    current = _body([_window()], payload=_payload(rows=[
        {"id": "b", "children": [{"id": "q"}, {"id": "p"}]},
    ]))
    projected, _, _ = _invoke(monkeypatch, current, runtime)
    stable = json.loads(next(row["content"] for row in projected["messages"]
                             if row.get("content", "").startswith("[StateFrame]\n")).split("\n", 1)[1])
    assert [row["id"] for row in stable["state"]["framefit5"]["rows"][0]["children"]] == ["q", "p"]
    original_frame = next(row["content"] for row in current["messages"]
                          if row.get("content", "").startswith("[StateFrame]\n"))
    assert materialize_stable_frame(projected["messages"]) == original_frame


def test_headroom_relays_to_current_windows_then_appends_again(monkeypatch):
    v1 = _window(path="pkg/a.py", revision=_revision("1"))
    v2 = _window(path="pkg/b.py", revision=_revision("2"))
    v3 = _window(path="pkg/c.py", revision=_revision("3"))
    other = _window(path="pkg/d.py", revision=_revision("4"))
    meter = SourceMeter({**{row["content"]: 4_000 for row in (v1, v2, v3)},
                         other["content"]: 1_000})
    runtime = SimpleNamespace(_stable_wire_state=None)
    first, _, events1 = _invoke(monkeypatch, _body([v1]), runtime, headroom=6_000, meter=meter)
    assert len(_messages_with_prefix(first["messages"], "[Current source:")) == 1
    assert any(row["kind"] == "stable_wire_epoch" for row in events1.events)

    # A different current file retains v1, but 4000 retained tokens fit in 6000.
    second, _, events2 = _invoke(monkeypatch, _body([v2]), runtime, headroom=6_000, meter=meter)
    second_sources = _messages_with_prefix(second["messages"], "[Current source:")
    assert second_sources == [v1, v2]
    assert not any(row["kind"] == "stable_wire_epoch" for row in events2.events)
    assert "Not current, do not edit from these:" in _status_rows(second["messages"])[0]["content"]

    # v1 and v2 are now both retained (8000 > 6000), so the headroom relayout fires. Keeping
    # both plus v3 is over the 10000 ceiling, so the longest fitting prefix keeps only v1.
    third, _, events3 = _invoke(monkeypatch, _body([v3]), runtime, headroom=6_000, meter=meter)
    assert _messages_with_prefix(third["messages"], "[Current source:") == [v1, v3]
    epochs = [row for row in events3.events if row["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0].get("reason") == "headroom"
    assert "\nEpoch 2\n" in _status_rows(third["messages"])[0]["content"]
    # The kept prefix holds only what fits the headroom, and the event reports that layout.
    assert epochs[0]["kept_prefix_windows"] == 1 and epochs[0]["retained_tokens"] == 4_000

    # The retained set now fits the headroom, so the next sends do not relayout again.
    again, _, events_again = _invoke(monkeypatch, _body([v3]), runtime, headroom=6_000, meter=meter)
    assert _messages_with_prefix(again["messages"], "[Current source:") == [v1, v3]
    assert not any(row["kind"] == "stable_wire_epoch" for row in events_again.events)
    _assert_stable_message_byte_prefix(third, again, rows=4)

    # A later new window appends after the delivered prefix, still in the same epoch.
    fourth, _, events4 = _invoke(monkeypatch, _body([v3, other]), runtime,
                                 headroom=6_000, meter=meter)
    assert _messages_with_prefix(fourth["messages"], "[Current source:") == [v1, v3, other]
    assert not any(row["kind"] == "stable_wire_epoch" for row in events4.events)
    _assert_stable_message_byte_prefix(third, fourth, rows=4)


def test_estimated_meter_keeps_same_epoch_stable_prefix_and_counts_retained_headroom(monkeypatch):
    windows, results, events, meter = _estimated_window_sequence(
        monkeypatch, source_method="estimate", extra_sets=([3],))

    # The first window is retained, not-current, on the next send. Its estimate
    # keeps the source block in epoch 1, so the serialized stable messages match.
    _assert_stable_message_byte_prefix(results[0], results[1])
    assert not any(event["kind"] == "stable_wire_epoch" for event in events[1])
    assert _status_rows(results[0]["messages"])[0]["content"].splitlines()[1] == "Epoch 1"
    assert "\nEpoch 1\n" in _status_rows(results[1]["messages"])[0]["content"]

    expected_sets = (
        ((windows[0]["content"],), 16_000),
        ((windows[0]["content"], windows[1]["content"]), 16_000),
        ((windows[0]["content"], windows[1]["content"],
          windows[2]["content"]), 16_000),
    )
    measured = {contents: (tokens, method)
                for contents, tokens, method in meter.source_measurements}
    assert measured[expected_sets[0][0]][0] < expected_sets[0][1]
    assert measured[expected_sets[1][0]][0] < expected_sets[1][1]
    retained_tokens, retained_method = measured[expected_sets[2][0]]
    assert retained_tokens > expected_sets[2][1]
    assert retained_method == "estimate"

    # Three retained windows exceed the headroom, so the relayout keeps the longest prefix
    # whose retained rows fit: windows 0 and 1. The event reports that chosen layout.
    epochs = [event for event in events[3] if event["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1
    assert epochs[0]["reason"] == "headroom"
    assert epochs[0]["epoch"] == 2
    assert epochs[0]["kept_prefix_windows"] == 2
    assert epochs[0]["retained_tokens"] == measured[expected_sets[1][0]][0]
    assert _sources(results[3]) == [windows[0], windows[1], windows[3]]
    _assert_stable_message_byte_prefix(results[2], results[3], rows=4)

    # The same retained set on the next send is within the headroom: no epoch.
    assert not any(event["kind"] == "stable_wire_epoch" for event in events[4])
    assert _sources(results[4]) == _sources(results[3])
    _assert_stable_message_byte_prefix(results[3], results[4])


def test_unavailable_source_meter_stops_the_kept_prefix_at_the_first_retained_row(monkeypatch):
    windows = [_window(path=f"pkg/{name}.py", revision=_revision(str(index)),
                       text="x" * 30_000 + "\n")
               for index, name in enumerate(("a", "b", "c", "d"), start=1)]
    runtime = SimpleNamespace(_stable_wire_state=None)
    meter = EstimatedMeter(source_method="unavailable")
    inputs = {"token_ceiling": 60_000, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}, "receipt": None}

    def send(current):
        return _invoke(monkeypatch, _body(current), runtime, headroom=16_000,
                       meter=meter, inputs=copy.deepcopy(inputs))

    def epochs_of(journal):
        return [event for event in journal.events if event["kind"] == "stable_wire_epoch"]

    first, _, _ = send(windows[:3])
    # Window c left. Its tokens cannot be measured, so the prefix stops right before it.
    second, _, journal = send(windows[:2])
    event = epochs_of(journal)[0]
    assert event["reason"] == "headroom" and event["epoch"] == 2
    assert event["kept_prefix_windows"] == 2
    assert _sources(second) == windows[:2]
    _assert_stable_message_byte_prefix(first, second, rows=4)
    assert all(method == "unavailable" and tokens == 0
               for _contents, tokens, method in meter.source_measurements)

    # No retained rows remain, so epochs stop bumping.
    third, _, journal = send(windows[:2])
    assert not epochs_of(journal)
    _assert_stable_message_byte_prefix(second, third)

    # A retained row at the very front keeps nothing: one more epoch, then stable again.
    fourth, _, journal = send([windows[1], windows[3]])
    event = epochs_of(journal)[0]
    assert event["epoch"] == 3 and event["kept_prefix_windows"] == 0
    assert _sources(fourth) == [windows[1], windows[3]]
    fifth, _, journal = send([windows[1], windows[3]])
    assert not epochs_of(journal)
    _assert_stable_message_byte_prefix(fourth, fifth)


def test_estimated_stable_request_uses_040b_reserve_and_fits_its_token_ceiling(monkeypatch):
    import kernel_cache_layout_candidate as candidate
    from kernel_context import token_estimate
    from kernel_stable_wire import is_stable_wire_projection

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", "16000")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda _plan, _messages, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")

    def body_with_optional_context():
        body = _final_layout_body()
        frame_index = next(index for index, row in enumerate(body["messages"])
                           if row.get("content", "").startswith("[StateFrame]\n"))
        body["messages"].insert(frame_index, {
            "role": "user", "content": "[Optional context]\n" + "x" * 20_000,
        })
        return body

    def render(body, *, meter, inputs, trim=None):
        if trim is not None:
            inputs["_final_layout_trim"] = trim
        return candidate.render_final_cache_candidate(
            SimpleNamespace(journal=Journal(), request_id="estimated-fit", step=1,
                            request_source_final_delivery=None),
            SimpleNamespace(_stable_wire_state=None), body, meter=meter, inputs=inputs)

    probe_meter = EstimatedMeter()
    probe = render(
        body_with_optional_context(), meter=probe_meter,
        inputs={"token_ceiling": 100_000, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert is_stable_wire_projection(probe["messages"])
    projected_tokens = token_estimate(probe)
    token_ceiling = projected_tokens - 128
    byte_ceiling = 1_000_000
    expected_reserve = candidate._final_layout_reserve(
        projected_tokens, token_ceiling,
        len(json.dumps(probe).encode("utf-8")), byte_ceiling)
    reserves = []

    def trim(token_reserve, byte_reserve):
        reserves.append((token_reserve, byte_reserve))
        assert token_reserve > 0
        original["messages"] = [
            row for row in original["messages"]
            if not str(row.get("content", "")).startswith("[Optional context]")
        ]
        return True

    original = body_with_optional_context()
    meter = EstimatedMeter()
    result = render(
        original, meter=meter,
        inputs={"token_ceiling": token_ceiling, "byte_ceiling": byte_ceiling,
                "source_plan": {}, "sources": {}, "receipt": None},
        trim=trim)

    assert reserves == [expected_reserve]
    assert all(not str(row.get("content", "")).startswith("[Optional context]")
               for row in original["messages"])
    measured_tokens, method = meter.measure(result)
    assert method == "estimate"
    assert measured_tokens <= token_ceiling
    assert token_estimate(result) == measured_tokens
    assert is_stable_wire_projection(result["messages"])


def test_fresh_epoch_that_still_does_not_fit_fails_open(monkeypatch):
    prior = _window(revision=_revision("1"), text="prior = 1\n")
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([prior]), runtime, headroom=6_000)
    huge = _window(revision=_revision("2"), text="large = 2\n" * 500)
    body = _body([huge])
    inputs = {"token_ceiling": 50, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}, "receipt": None}
    result, _, journal = _invoke(monkeypatch, body, runtime, headroom=6_000, inputs=inputs)
    assert result is body
    assert any(row["kind"] == "stable_wire_failed_open" for row in journal.events)


def test_request_envelope_relayout_uses_current_windows_then_appends(monkeypatch):
    class EnvelopeMeter:
        def measure(self, body):
            messages = body.get("messages", [])
            sources = [row for row in messages
                       if str(row.get("content", "")).startswith("[Current source:")]
            if len(sources) == len(messages):
                return len(sources), "tokenize"
            return (10_001 if len(sources) > 1 else 1), "tokenize"

    meter = EnvelopeMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    first = _window(revision=_revision("1"))
    second = _window(revision=_revision("2"), text="value = 2\n")
    _invoke(monkeypatch, _body([first]), runtime, headroom=6_000, meter=meter)

    # Planted control: retained source messages consume this same prompt envelope.
    retained_plus_current = _body([first, second])
    assert meter.measure(retained_plus_current)[0] > 10_000

    projected, _, journal = _invoke(
        monkeypatch, _body([second]), runtime, headroom=6_000, meter=meter)
    assert _messages_with_prefix(projected["messages"], "[Current source:") == [second]
    assert meter.measure(projected)[0] <= 10_000
    assert "\nEpoch 2\n" in _status_rows(projected["messages"])[0]["content"]
    assert any(row["kind"] == "stable_wire_epoch" and row.get("action") == "relayout"
               for row in journal.events)


def test_fresh_epoch_that_still_exceeds_request_envelope_fails_open(monkeypatch):
    class EnvelopeMeter:
        def measure(self, body):
            messages = body.get("messages", [])
            sources = [row for row in messages
                       if str(row.get("content", "")).startswith("[Current source:")]
            if len(sources) == len(messages):
                return len(sources), "tokenize"
            frame = next((row.get("content", "") for row in messages
                          if str(row.get("content", "")).startswith("[StateFrame]\n")), "")
            return (10_001 if len(frame) > 5_000 else 1), "tokenize"

    meter = EnvelopeMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    prior = _window(revision=_revision("1"))
    _invoke(monkeypatch, _body([prior]), runtime, headroom=6_000, meter=meter)
    payload = _payload()
    payload["state"]["framefit5"]["stable_large_value"] = "x" * 6_000
    current = _window(revision=_revision("2"), text="value = 2\n")
    body = _body([current], payload=payload)

    result, _, journal = _invoke(monkeypatch, body, runtime, headroom=6_000, meter=meter)
    assert result is body
    assert any(row["kind"] == "stable_wire_failed_open"
               and row.get("reason") == "fresh_epoch_exceeds_measured_envelope"
               for row in journal.events)


@pytest.mark.parametrize("change", ("task", "editable", "tools", "profile_switches"))
def test_structural_input_change_starts_one_current_only_epoch(monkeypatch, change):
    first_window = _window(path="pkg/old.py", revision=_revision("1"))
    current = _window(path="pkg/new.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    first_body = _body([first_window])
    first_inputs = {"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                    "source_plan": {"editable_paths": ["pkg/old.py"]}, "sources": {},
                    "receipt": None, "profile": "default", "switch_set": {"X": "on"}}
    _invoke(monkeypatch, first_body, runtime, inputs=copy.deepcopy(first_inputs))

    next_body = _body([current])
    next_inputs = copy.deepcopy(first_inputs)
    if change == "task":
        next_body["messages"][1]["content"] = "Task: a structurally new task"
    elif change == "editable":
        next_inputs["source_plan"]["editable_paths"] = ["pkg/new.py"]
    elif change == "tools":
        next_body["tools"] = [{"type": "function", "function": {"name": "write_file"}}]
    else:
        next_inputs["profile"] = "strict"
        next_inputs["switch_set"]["X"] = "off"

    projected, _, journal = _invoke(monkeypatch, next_body, runtime, inputs=next_inputs)
    assert _messages_with_prefix(projected["messages"], "[Current source:") == [current]
    epochs = [event for event in journal.events if event["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0].get("reason") == "structural"
    assert "\nEpoch 2\n" in _status_rows(projected["messages"])[0]["content"]


def test_structural_relayout_keeps_current_windows_in_run_first_seen_order(monkeypatch):
    first = _window(path="pkg/a.py", revision=_revision("1"))
    second = _window(path="pkg/b.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([first, second]), runtime)

    rebased_body = _body([second, first])
    rebased_body["messages"][1]["content"] = "Task: changed task"
    projected, _, journal = _invoke(monkeypatch, rebased_body, runtime)
    assert _messages_with_prefix(projected["messages"], "[Current source:") == [first, second]
    epochs = [event for event in journal.events if event["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0].get("reason") == "structural"


def test_stable_frame_change_keeps_epoch_and_appends_windows_after_delivered_ones(monkeypatch):
    """The frame sits after the windows, so a frame-only change must not relayout them."""
    old = _window(path="pkg/old.py", revision=_revision("1"))
    current = _window(path="pkg/current.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    first, _, _ = _invoke(monkeypatch, _body([old]), runtime)
    first_status = _status_rows(first["messages"])[0]["content"]
    assert "\nEpoch 1\n" in first_status

    payload = _payload()
    payload["state"]["framefit5"]["stable_marker"] = "changed"
    projected, _, journal = _invoke(monkeypatch, _body([current], payload=payload), runtime)
    assert not [event for event in journal.events if event["kind"] == "stable_wire_epoch"]
    assert _messages_with_prefix(projected["messages"], "[Current source:") == [old, current]
    status = _status_rows(projected["messages"])[0]["content"]
    assert "\nEpoch 1\n" in status
    assert materialize_stable_frame(projected["messages"]) != materialize_stable_frame(first["messages"])


def test_headroom_guard_rejects_values_below_6000_at_first_request(monkeypatch):
    body = _body([_window()])
    runtime = SimpleNamespace(_stable_wire_state=None)
    with pytest.raises(
            ValueError,
            match=r"STATEFRAME_STABLE_WIRE_HEADROOM.*compactions every 2-3 requests erase the gain"):
        _invoke(monkeypatch, body, runtime, headroom=5_999)
    accepted, _, _ = _invoke(monkeypatch, body, runtime, headroom=6_000)
    assert accepted is not body


def test_off_matches_unset_and_does_not_add_checkpoint_fields(monkeypatch):
    from kernel_continuation_restart import runtime_fields
    from kernel_cache_layout_candidate import render_final_cache_candidate

    body = _body([_window()])
    runtime = SimpleNamespace()
    monkeypatch.delenv("STATEFRAME_STABLE_WIRE", raising=False)
    unset = project_stable_wire(None, runtime, body, meter=None, inputs=None)
    fields_unset = runtime_fields(runtime)
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    off = project_stable_wire(None, runtime, body, meter=None, inputs=None)
    fields_off = runtime_fields(runtime)
    assert off is body and unset is body
    assert json.dumps(off, ensure_ascii=False).encode() == json.dumps(unset, ensure_ascii=False).encode()
    assert fields_off == fields_unset

    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _receipt: True)
    layout_input = _final_layout_body()
    inputs = {"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}, "receipt": None}
    monkeypatch.delenv("STATEFRAME_STABLE_WIRE", raising=False)
    without_switch = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, layout_input,
        meter=Meter(), inputs=copy.deepcopy(inputs))
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    with_switch_off = render_final_cache_candidate(
        SimpleNamespace(request_source_final_delivery=None), None, layout_input,
        meter=Meter(), inputs=copy.deepcopy(inputs))
    assert json.dumps(with_switch_off, ensure_ascii=False).encode() == \
        json.dumps(without_switch, ensure_ascii=False).encode()


def test_detached_runtime_carries_stable_wire_state_only_when_enabled(monkeypatch):
    from kernel_request_candidate import detached_context, runtime_authority_fields

    state = {"schema": "stateframe-stable-wire/1", "epoch": 1}
    runtime = SimpleNamespace(_stable_wire_state=state, owner=SimpleNamespace())
    monkeypatch.delenv("STATEFRAME_STABLE_WIRE", raising=False)
    assert "_stable_wire_state" not in runtime_authority_fields(runtime)

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    assert "_stable_wire_state" in runtime_authority_fields(runtime)
    candidate_capture, candidate_runtime = detached_context(SimpleNamespace(), runtime)
    assert candidate_runtime._stable_wire_state == state
    candidate_runtime._stable_wire_state["epoch"] = 2
    runtime.__dict__.update({
        name: copy.deepcopy(getattr(candidate_runtime, name))
        for name in runtime_authority_fields(candidate_runtime)
        if hasattr(candidate_runtime, name)
    })
    assert runtime._stable_wire_state["epoch"] == 2
    assert candidate_capture.completion_owner_runtime is candidate_runtime


def test_stable_wire_requires_final_cache_layout_and_numeric_headroom(monkeypatch):
    from stateframe import runner
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    with pytest.raises(ValueError, match="STATEFRAME_STABLE_WIRE.*STATEFRAME_FINAL_CACHE_LAYOUT"):
        render_final_cache_candidate(None, None, {}, meter=None, inputs=None)

    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, "engine_switches", lambda _directory=runner.ENGINE_DIR: frozenset(
        {"STATEFRAME_STABLE_WIRE_HEADROOM"}))
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", "6000")
    assert runner.resolve_opt_in_switches()["STATEFRAME_STABLE_WIRE_HEADROOM"] == "6000"
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", "0")
    with pytest.raises(runner.RunError, match="positive whole number"):
        runner.resolve_opt_in_switches()


def test_headroom_switch_is_documented_as_a_16000_token_retention_limit():
    from pathlib import Path

    guide = Path(__file__).resolve().parents[1] / "docs/guide/running.md"
    text = guide.read_text(encoding="utf-8")
    assert "STATEFRAME_STABLE_WIRE_HEADROOM" in text
    assert "16000" in text
    assert "compactions every 2-3 requests erase the gain" in text


def test_stable_wire_events_are_registered_and_counted():
    from kernel_protocol_spec import _event_rows
    from kernel_run_vitals import RunVitalsAccumulator

    rows = {row.get("kind"): row for row in _event_rows()}
    assert {"stable_wire_epoch", "stable_wire_failed_open"} <= rows.keys()
    vitals = RunVitalsAccumulator()
    vitals.observe({"kind": "stable_wire_epoch"})
    vitals.observe({"kind": "stable_wire_failed_open"})
    snapshot = vitals.snapshot()
    assert snapshot["stable_wire_epochs"] == 1
    assert snapshot["cache_layout_fail_open"] == 1


def test_checkpoint_round_trip_preserves_next_projection_bytes(monkeypatch):
    from kernel_continuation_restart import restore_runtime_field_values, runtime_fields

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setattr("kernel_continuation_restart.RUNTIME_FIELDS", ("sentinel",))
    first = _window(revision=_revision("1"))
    second = _window(revision=_revision("2"))
    runtime = SimpleNamespace(sentinel=None, _stable_wire_state=None)
    _invoke(monkeypatch, _body([first]), runtime)
    assert "_stable_wire_state" in runtime_fields(runtime)
    saved = {field: copy.deepcopy(getattr(runtime, field)) for field in runtime_fields(runtime)}
    resumed = SimpleNamespace(sentinel=None, _stable_wire_state=None)
    restore_runtime_field_values(resumed, saved)

    after_restart, _, _ = _invoke(monkeypatch, _body([second]), resumed)
    uninterrupted, _, _ = _invoke(monkeypatch, _body([second]), runtime)
    assert json.dumps(after_restart, ensure_ascii=False, separators=(",", ":")).encode() == \
        json.dumps(uninterrupted, ensure_ascii=False, separators=(",", ":")).encode()


def _render_final_layout_stable(monkeypatch, body=None, *, runtime=None, meter=None, inputs=None):
    from kernel_cache_layout_candidate import render_final_cache_candidate

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE_HEADROOM", "16000")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda _plan, _messages, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    body = copy.deepcopy(body if body is not None else _final_layout_body())
    runtime = runtime or SimpleNamespace(_stable_wire_state=None)
    capture = SimpleNamespace(journal=Journal(), request_id="layout-stable", step=1,
                              request_source_final_delivery=None)
    request_inputs = inputs or {
        "token_ceiling": 10_000, "byte_ceiling": 1_000_000,
        "source_plan": {}, "sources": {}, "receipt": None,
    }
    result = render_final_cache_candidate(
        capture, runtime, body, meter=meter or Meter(), inputs=request_inputs)
    return body, result, capture, runtime, request_inputs


def _assert_final_layout_and_stable_wire(body, record):
    from kernel_cache_layout_view import FINAL_LAYOUT_NAME
    from kernel_stable_wire import is_stable_wire_projection

    assert body["messages"][-1].get("name") == FINAL_LAYOUT_NAME
    assert is_stable_wire_projection(body["messages"])
    assert record["final_layout"] == "applied"
    assert record["stable_wire"] == "applied"


def _assert_plain_wire(body):
    from kernel_stable_wire import is_stable_wire_projection

    assert not is_stable_wire_projection(body["messages"])


def test_stable_wire_applies_over_the_default_on_final_layout(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    _input, result, capture, _runtime, _inputs = _render_final_layout_stable(monkeypatch)
    _assert_final_layout_and_stable_wire(result, capture._final_cache_layout_record)

    # Plant an identity final-layout renderer. The assertion must reject the
    # stable projection when its required final authority layout is absent.
    with monkeypatch.context() as planted:
        planted.setattr(candidate, "_checked_final_cache_projection", lambda body: body)
        _body, control, control_capture, _runtime, _inputs = _render_final_layout_stable(planted)
        with pytest.raises(AssertionError):
            _assert_final_layout_and_stable_wire(control, control_capture._final_cache_layout_record)


def test_stable_wire_refuses_an_explicitly_disabled_final_layout(monkeypatch):
    import kernel_stable_wire

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")

    def assert_refused():
        from kernel_cache_layout_candidate import render_final_cache_candidate
        try:
            render_final_cache_candidate(None, None, {}, meter=None, inputs=None)
        except ValueError as error:
            assert "STATEFRAME_STABLE_WIRE" in str(error)
            assert "STATEFRAME_FINAL_CACHE_LAYOUT" in str(error)
        else:
            raise AssertionError("stable wire accepted an explicitly disabled final layout")

    assert_refused()

    # Plant the old silent-off behavior so the refusal assertion is known to
    # catch a missing stable-wire guard.
    with monkeypatch.context() as planted:
        planted.setattr(kernel_stable_wire, "mode", lambda: "off")
        with pytest.raises(AssertionError):
            assert_refused()


def test_stable_wire_failed_open_reason_is_recorded_and_sends_the_layout_body(monkeypatch, tmp_path):
    from kernel_stable_wire import is_stable_wire_projection

    class StableOverflowMeter:
        def measure(self, body):
            stable = is_stable_wire_projection(body.get("messages", []))
            return (1000 if stable else 100), "tokenize"

    _input, result, capture, _runtime, _inputs = _render_final_layout_stable(
        monkeypatch, meter=StableOverflowMeter(),
        inputs={"token_ceiling": 200, "byte_ceiling": 1_000_000,
                "source_plan": {}, "sources": {}, "receipt": None})

    def assert_recorded_fail_open(body, record):
        assert record["final_layout"] == "applied"
        assert record.get("stable_wire") == "failed_open"
        assert not is_stable_wire_projection(body["messages"])

    assert_recorded_fail_open(result, capture._final_cache_layout_record)
    assert any(event["kind"] == "stable_wire_failed_open" for event in capture.journal.events)

    import kernel_bounded_http
    from kernel_measurement import ACTIVE_REQUEST
    from stateframe import _child
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    run = tmp_path / "failed-open-send"
    run.mkdir()
    _child.install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions")
    active = ACTIVE_REQUEST.set(capture)
    try:
        kernel_bounded_http.request_bytes(
            json.dumps(result).encode(), "http://127.0.0.1:8000/v1/chat/completions", "")
    finally:
        ACTIVE_REQUEST.reset(active)
    sent = json.loads((run / "sends.jsonl").read_text(encoding="utf-8"))
    assert sent["final_layout"] == "applied"
    assert sent["stable_wire"] == "failed_open"
    reason = "fresh_epoch_exceeds_measured_envelope"
    assert capture._final_cache_layout_record["stable_wire_reason"] == reason

    def assert_recorded_reason(record):
        assert record.get("stable_wire_reason") == reason

    assert_recorded_reason(sent)

    # Plant the prior recorder defect: a missing reason must fail this check.
    missing_reason = dict(sent)
    missing_reason.pop("stable_wire_reason")
    with pytest.raises(AssertionError):
        assert_recorded_reason(missing_reason)

    # Plant the former unrecorded fail-open. The send remains layout-shaped,
    # but the disposition assertion must reject it.
    import kernel_stable_wire
    with monkeypatch.context() as planted:
        planted.setattr(kernel_stable_wire, "_record_disposition", lambda *_args, **_kwargs: None)
        _body, control, control_capture, _runtime, _inputs = _render_final_layout_stable(
            planted, meter=StableOverflowMeter(),
            inputs={"token_ceiling": 200, "byte_ceiling": 1_000_000,
                    "source_plan": {}, "sources": {}, "receipt": None})
        with pytest.raises(AssertionError):
            assert_recorded_fail_open(control, control_capture._final_cache_layout_record)


def test_stable_wire_reserves_040b_projection_overhead_before_fail_open(monkeypatch):
    import kernel_cache_layout_candidate as candidate
    from kernel_stable_wire import is_stable_wire_projection

    class TrimAwareMeter:
        def __init__(self):
            self.trimmed = False

        def measure(self, body):
            stable = is_stable_wire_projection(body.get("messages", []))
            return (120 if stable and self.trimmed else 180 if stable else 100), "tokenize"

    def render(meter, inputs):
        reserves = []

        def trim(tokens, byte_count):
            reserves.append((tokens, byte_count))
            if tokens >= 46:
                meter.trimmed = True
            return True

        inputs["_final_layout_trim"] = trim
        body, result, capture, _runtime, _inputs = _render_final_layout_stable(
            monkeypatch, meter=meter, inputs=inputs)
        return body, result, capture, reserves, meter

    inputs = {"token_ceiling": 150, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}, "receipt": None}
    _input, result, capture, reserves, measured = render(TrimAwareMeter(), inputs)
    _assert_final_layout_and_stable_wire(result, capture._final_cache_layout_record)
    assert reserves == [(46, 256)]
    assert measured.measure(result)[0] <= 150

    # Plant a zero reserve. Its callback cannot free the measured 30-token
    # overage, and the stable projection assertion must catch the fail-open.
    with monkeypatch.context() as planted:
        planted.setattr(candidate, "_final_layout_reserve", lambda *_args: (0, 0))
        _body, control, control_capture, _reserves, _measured = render(
            TrimAwareMeter(), {"token_ceiling": 150, "byte_ceiling": 1_000_000,
                              "source_plan": {}, "sources": {}, "receipt": None})
        with pytest.raises(AssertionError):
            _assert_final_layout_and_stable_wire(control, control_capture._final_cache_layout_record)


def test_resume_recorded_plain_stable_wire_send_does_not_project(monkeypatch, tmp_path):
    import kernel_bounded_http
    import kernel_cache_layout_candidate as candidate
    from stateframe import _child
    from stateframe.replay_tape import Replayer
    from kernel_stable_wire import is_stable_wire_projection

    # Record a final-layout request produced with stable wire off.
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "off")
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda _plan, _messages, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    from kernel_cache_layout_candidate import render_final_cache_candidate
    recorded = render_final_cache_candidate(
        SimpleNamespace(journal=Journal(), request_id="recorded", step=1,
                        request_source_final_delivery=None), None, _final_layout_body(),
        meter=Meter(), inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                               "source_plan": {}, "sources": {}, "receipt": None})
    recorded_wire = json.dumps(recorded).encode()
    request_path = tmp_path / "recorded.request.json"
    response_path = tmp_path / "recorded.response.json"
    response_path.write_bytes(b"{}")
    request_path.write_bytes(recorded_wire)
    replay = Replayer([{"send": 1, "request": str(request_path), "response": str(response_path),
                        "response_sha256": hashlib.sha256(b"{}").hexdigest()}])

    monkeypatch.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", None, raising=False)
    monkeypatch.setattr(_child, "HOST_STATE", dict(_child.HOST_STATE))
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", kernel_bounded_http.request_bytes)
    run = tmp_path / "replay-run"
    run.mkdir()
    _child.install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions", replay)

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    rebuilt = render_final_cache_candidate
    rebuilt_body = _final_layout_body()
    capture = SimpleNamespace(journal=Journal(), request_id="replay", step=1,
                              request_source_final_delivery=None)
    result = rebuilt(
        capture, SimpleNamespace(_stable_wire_state=None), rebuilt_body,
        meter=Meter(), inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                               "source_plan": {}, "sources": {}, "receipt": None})
    assert json.dumps(result).encode() == recorded_wire
    assert replay.answer(recorded_wire) == b"{}"
    assert not is_stable_wire_projection(result["messages"])
    assert capture._final_cache_layout_record["stable_wire"] == "plain"

    # Plant a missing replay decision provider. Current opt-in then projects,
    # and the plain-wire assertion must reject the changed request.
    with monkeypatch.context() as planted:
        planted.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", None)
        planted.setattr(candidate, "_REPLAY_LAYOUT_DECISION_PROVIDER", lambda: "layout")
        _body = _final_layout_body()
        control = rebuilt(
            SimpleNamespace(journal=Journal(), request_id="control", step=1,
                            request_source_final_delivery=None),
            SimpleNamespace(_stable_wire_state=None), _body, meter=Meter(),
            inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                    "source_plan": {}, "sources": {}, "receipt": None})
        with pytest.raises(AssertionError):
            _assert_plain_wire(control)


def test_metadata_rerender_of_delivered_window_starts_fresh_epoch(monkeypatch):
    # Live control-arm crash: same path/revision/range, unchanged source lines, but the
    # window's delivery metadata re-rendered. That needs a relayout, not a run abort.
    first = _window(path="pkg/a.py", revision=_revision("1"))
    other = _window(path="pkg/b.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([first, other]), runtime)

    rerendered = copy.deepcopy(first)
    rerendered["content"] = rerendered["content"].replace(
        '"max_lines":1000', '"max_lines":999')
    assert rerendered["content"] != first["content"]
    projected, _, journal = _invoke(monkeypatch, _body([rerendered, other]), runtime)

    sources = _messages_with_prefix(projected["messages"], "[Current source:")
    assert sources == [rerendered, other]
    epochs = [event for event in journal.events if event["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0].get("reason") == "rerender"
    assert "\nEpoch 2\n" in _status_rows(projected["messages"])[0]["content"]

    # The next send keeps the rerendered epoch's prefix byte-stable.
    again, _, journal = _invoke(monkeypatch, _body([rerendered, other]), runtime)
    assert _messages_with_prefix(again["messages"], "[Current source:") == [rerendered, other]
    assert not [event for event in journal.events if event["kind"] == "stable_wire_epoch"]


def test_changed_source_lines_under_same_key_relayout_instead_of_raising(monkeypatch):
    """A delivered window whose source lines re-rendered under the same
    path/revision/ranges used to raise StableWireError and end the run.
    It is renderer drift, not a file edit, so it now gets
    the same answer as metadata drift: a rerender relayout in a fresh epoch that
    sends the current bytes and keeps only the prefix ahead of the changed row."""
    first = _window(path="pkg/a.py", revision=_revision("1"), text="value = 1\n")
    other = _window(path="pkg/b.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([other, first]), runtime)
    changed = copy.deepcopy(first)
    changed["content"] = changed["content"].replace("value = 1", "value = 9")
    assert changed["content"] != first["content"]

    projected, _, journal = _invoke(monkeypatch, _body([other, changed]), runtime)
    assert _sources(projected) == [other, changed]
    assert first not in projected["messages"]
    _only_relayout(journal, "rerender")
    event = next(e for e in journal.events if e["kind"] == "stable_wire_epoch")
    assert event["epoch"] == 2 and event["kept_prefix_windows"] == 1
    assert "\nEpoch 2\n" in _status_rows(projected["messages"])[0]["content"]

    # The next send keeps the new epoch's prefix byte-stable.
    again, _, journal = _invoke(monkeypatch, _body([other, changed]), runtime)
    assert _sources(again) == [other, changed]
    assert not [e for e in journal.events if e["kind"] == "stable_wire_epoch"]


def test_structural_rebase_skips_the_rerender_check(monkeypatch):
    """A structural rebase discards the previous layout, so a row whose source lines
    re-rendered under the same key must not be compared against it (body 8)."""
    first = _window(path="pkg/a.py", revision=_revision("1"), text="value = 1\n")
    other = _window(path="pkg/b.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([first, other]), runtime)
    changed = copy.deepcopy(first)
    changed["content"] = changed["content"].replace("value = 1", "value = 9")

    rebased = _body([other, changed])
    rebased["messages"][1]["content"] = "Task: a structurally new task"
    projected, _, journal = _invoke(monkeypatch, rebased, runtime)
    assert _sources(projected) == [changed, other]
    assert first not in projected["messages"]
    _only_relayout(journal, "structural")
    event = next(e for e in journal.events if e["kind"] == "stable_wire_epoch")
    assert event["epoch"] == 2 and event["kept_prefix_windows"] == 0


# --- prefix-preserving relayout -------------------------------------------------------


def _relayout_windows():
    """A-D form the previous layout; Z was seen first but dropped, E and F are new."""
    return {name: _window(path=f"pkg/{name}.py", revision=_revision(str(index)))
            for index, name in enumerate("ABCDEFZ", start=1)}


def _late_z_layout(monkeypatch, w, runtime, **kwargs):
    """Leave the run with layout [A, B, C, D, Z] while Z is first in first-seen order.

    That is the live shape: a window that dropped out of one layout and came back is
    appended last, so the previous layout order differs from run first-seen order.
    """
    _invoke(monkeypatch, _body([w["Z"], w["A"], w["B"], w["C"], w["D"]]), runtime, **kwargs)
    changed_task = _body([w["A"], w["B"], w["C"], w["D"]])
    changed_task["messages"][1]["content"] = "Task: a structurally new task"
    _invoke(monkeypatch, changed_task, runtime, **kwargs)
    again = _body([w["A"], w["B"], w["C"], w["D"], w["Z"]])
    again["messages"][1]["content"] = "Task: a structurally new task"
    projected, _, _ = _invoke(monkeypatch, again, runtime, **kwargs)
    assert _sources(projected) == [w[n] for n in "ABCDZ"]


def _sources(result):
    return _messages_with_prefix(result["messages"], "[Current source:")


def _only_relayout(journal, reason):
    epochs = [event for event in journal.events if event["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0]["action"] == "relayout" and epochs[0]["reason"] == reason


def _relayout_body(windows):
    body = _body(windows)
    body["messages"][1]["content"] = "Task: a structurally new task"
    return body


def test_rerender_relayout_keeps_unchanged_prefix_then_new_content_then_first_seen(monkeypatch):
    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _late_z_layout(monkeypatch, w, runtime)

    rerendered = copy.deepcopy(w["C"])
    rerendered["content"] = rerendered["content"].replace('"max_lines":1000', '"max_lines":999')
    # D dropped; F is shown before E, so F is first seen first.
    current = [w["F"], w["E"], w["Z"], rerendered, w["B"], w["A"]]
    projected, _, journal = _invoke(monkeypatch, _relayout_body(current), runtime)

    delivered = _sources(projected)
    assert delivered[:2] == [w["A"], w["B"]]            # (a) unchanged prefix, same order
    assert delivered.count(rerendered) == 1 and w["C"] not in delivered  # (b) new content, once
    assert w["D"] not in delivered                       # (c) dropped window is gone
    # (d) everything after the kept prefix follows first-seen order, new windows last.
    assert delivered[2:] == [w["Z"], rerendered, w["F"], w["E"]]
    _only_relayout(journal, "rerender")


def test_headroom_relayout_keeps_unchanged_prefix_then_first_seen(monkeypatch):
    w = _relayout_windows()
    costs = {row["content"]: 500 for row in w.values()}
    costs[w["C"]["content"]] = 7_000  # the dropped window alone exceeds the 6000 headroom
    meter = SourceMeter(costs)
    runtime = SimpleNamespace(_stable_wire_state=None)
    _late_z_layout(monkeypatch, w, runtime, headroom=6_000, meter=meter)

    # C dropped, so D (current, unchanged) no longer continues the prefix and moves behind it.
    current = [w["F"], w["E"], w["D"], w["Z"], w["B"], w["A"]]
    projected, _, journal = _invoke(monkeypatch, _relayout_body(current), runtime,
                                    headroom=6_000, meter=meter)
    delivered = _sources(projected)
    assert delivered[:2] == [w["A"], w["B"]]
    assert w["C"] not in delivered
    assert delivered.count(w["D"]) == 1
    assert delivered[2:] == [w["Z"], w["D"], w["F"], w["E"]]
    _only_relayout(journal, "headroom")


def test_envelope_relayout_keeps_unchanged_prefix_then_first_seen(monkeypatch):
    class EnvelopeMeter:
        def measure(self, body):
            messages = body.get("messages", [])
            sources = [row for row in messages
                       if str(row.get("content", "")).startswith("[Current source:")]
            if len(sources) == len(messages):
                return len(sources), "tokenize"
            return (10_001 if len(sources) > 6 else 1), "tokenize"

    meter = EnvelopeMeter()
    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _late_z_layout(monkeypatch, w, runtime, meter=meter)

    # Retaining C on top of the six current windows would be seven sources: over the envelope.
    current = [w["F"], w["E"], w["D"], w["Z"], w["B"], w["A"]]
    projected, _, journal = _invoke(monkeypatch, _relayout_body(current), runtime,
                                    meter=meter)
    delivered = _sources(projected)
    assert delivered[:2] == [w["A"], w["B"]]
    assert w["C"] not in delivered and delivered.count(w["D"]) == 1
    assert delivered[2:] == [w["Z"], w["D"], w["F"], w["E"]]
    assert meter.measure(projected)[0] <= 10_000
    _only_relayout(journal, "envelope")


def test_relayout_order_keeps_the_prefix_and_sorts_the_rest_first_seen():
    from kernel_stable_wire import _relayout_order, _source_tokens, _window_entries
    from kernel_stable_wire_ownership import request_ownership

    w = _relayout_windows()
    messages = _body([w["F"], w["E"], w["D"], w["B"], w["A"]])["messages"]
    entries = _window_entries(messages, request_ownership(messages, len(messages) - 2))
    by_path = {row["identity"]["path"]: row for row in entries}
    previous = [by_path["pkg/A.py"], by_path["pkg/B.py"], {
        **by_path["pkg/D.py"], "key": "stale", "content": "gone"}]
    rank = {row["key"]: index for index, row in enumerate(reversed(entries))}
    plain = sorted(copy.deepcopy(entries), key=lambda row: rank[row["key"]])
    meter = SourceMeter({row["content"]: 123 for row in entries})

    ordered = _relayout_order(previous, entries, rank, 2)
    assert sorted(ordered, key=lambda row: row["key"]) == sorted(plain, key=lambda row: row["key"])
    assert _source_tokens(meter, [r["message"] for r in ordered]) == \
        _source_tokens(meter, [r["message"] for r in plain])
    assert [r["identity"]["path"] for r in ordered[:2]] == ["pkg/A.py", "pkg/B.py"]

    # keep=3 retains the stale (no longer current) third row and the other current rows follow
    # it in first-seen order; keep=0 is the plain first-seen sort.
    retained = _relayout_order(previous, entries, rank, 3)
    assert [r["key"] for r in retained[:3]] == [previous[0]["key"], previous[1]["key"], "stale"]
    assert retained[2]["content"] == "gone"
    rest = [row for row in plain if row["key"] not in {previous[0]["key"], previous[1]["key"]}]
    assert [r["key"] for r in retained[3:]] == [r["key"] for r in rest]
    assert _relayout_order(previous, entries, rank, 0) == plain


# --- longest-fitting-prefix relayout (class killer) -----------------------------------


class ForcedRelayoutMeter:
    """Sources-only bodies count windows. A full request is over the envelope while
    ``reject_next`` is set (one shot, consumed by the append-only fit check) or when it
    holds more than ``max_sources`` windows. This forces an envelope relayout."""

    def __init__(self, max_sources=99):
        self.max_sources = max_sources
        self.reject_next = False
        self.calls = 0
        self.full_counts = []  # window count of every full-request measurement, in order

    def measure(self, body):
        messages = body.get("messages", [])
        sources = [row for row in messages
                   if str(row.get("content", "")).startswith("[Current source:")]
        if len(sources) == len(messages):
            return len(sources), "tokenize"
        self.calls += 1
        self.full_counts.append(len(sources))
        if self.reject_next:
            self.reject_next = False
            return 10_001, "tokenize"
        return (10_001 if len(sources) > self.max_sources else 1), "tokenize"


def _epochs(journal):
    return [event for event in journal.events if event["kind"] == "stable_wire_epoch"]


def _fake_tokens(messages):
    """Token-level view: one fake token per character of concatenated content."""
    return list("".join(str(row.get("content", "")) for row in messages))


def test_envelope_relayout_keeps_whole_prefix_when_a_window_leaves_and_a_new_one_arrives(monkeypatch):
    w = _relayout_windows()
    meter = ForcedRelayoutMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    # Z is seen first but is not delivered in send N, so first-seen order differs from the
    # append-only order and the relayout candidate is a different layout.
    _invoke(monkeypatch, _body([w[n] for n in "ZABCD"]), runtime, meter=meter)
    first, _, _ = _invoke(monkeypatch, _relayout_body([w[n] for n in "ABCD"]), runtime,
                          meter=meter)
    assert _sources(first) == [w[n] for n in "ABCD"]

    meter.reject_next = True
    # A left the current set, E is new, Z returns, and the upstream order is shuffled.
    second, _, journal = _invoke(
        monkeypatch, _relayout_body([w["E"], w["D"], w["C"], w["B"], w["Z"]]), runtime,
        meter=meter)

    assert _sources(second) == [w[n] for n in "ABCDZE"]
    head = 2 + 4  # system, task, windows A-D
    assert second["messages"][:head] == first["messages"][:head]   # identical message dicts
    _assert_stable_message_byte_prefix(first, second, rows=head)   # identical serialized bytes
    assert _fake_tokens(second["messages"][:head]) == _fake_tokens(first["messages"][:head])
    event = _epochs(journal)[0]
    assert event["reason"] == "envelope" and event["action"] == "relayout"
    assert event["kept_prefix_windows"] == 4
    assert event["retained_tokens"] == 1  # A is the one retained, not-current window
    status = _status_rows(second["messages"])[0]["content"]
    current_part, retained_part = status.split("Not current, do not edit from these:")
    assert "pkg/A.py: current" not in current_part and "pkg/A.py" in retained_part
    assert "pkg/E.py: current" in current_part


def test_envelope_relayout_keeps_only_the_prefix_that_fits(monkeypatch):
    w = _relayout_windows()
    meter = ForcedRelayoutMeter(max_sources=4)
    runtime = SimpleNamespace(_stable_wire_state=None)
    first, _, _ = _invoke(monkeypatch, _body([w[n] for n in "ABCD"]), runtime, meter=meter)

    meter.reject_next = True
    # C and D left. Keeping k=2 gives A,B,F,E (4 windows); k=3 or k=4 is over the envelope.
    projected, _, journal = _invoke(
        monkeypatch, _body([w["F"], w["E"], w["A"], w["B"]]), runtime, meter=meter)
    assert _sources(projected) == [w["A"], w["B"], w["F"], w["E"]]
    assert _epochs(journal)[0]["kept_prefix_windows"] == 2
    _assert_stable_message_byte_prefix(first, projected, rows=2 + 2)


def test_binary_search_uses_few_meter_calls(monkeypatch):
    import kernel_stable_wire

    names = [f"w{i}" for i in range(16)]
    w = {n: _window(path=f"pkg/{n}.py", revision=_revision("a"), text=f"{n} = 1\n")
         for n in names}
    meter = ForcedRelayoutMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    first, _, _ = _invoke(monkeypatch, _body([w[n] for n in names[:8]]), runtime, meter=meter)
    meter.max_sources = 5
    meter.calls = 0
    meter.full_counts.clear()
    proofs = []
    real_prove = kernel_stable_wire._prove_projection
    monkeypatch.setattr(kernel_stable_wire, "_prove_projection",
                        lambda *args, **kwargs: proofs.append(1) or real_prove(*args, **kwargs))
    # Five of the eight prior windows left. The append-only layout (8 windows) is measured
    # once and rejected. Keeping all 8 would be the same bytes, so it is not measured again.
    # The search then probes k=4 (fits), k=6 (over), k=5 (fits) and reuses that last report.
    projected, _, journal = _invoke(monkeypatch, _body([w[n] for n in names[:3]]), runtime,
                                    meter=meter)
    assert _sources(projected) == [w[n] for n in names[:5]]
    assert _epochs(journal)[0]["kept_prefix_windows"] == 5
    assert meter.full_counts == [8, 4, 6, 5]
    # Only the append-only candidate and the chosen layout get the full proof.
    assert len(proofs) == 2
    _assert_stable_message_byte_prefix(first, projected, rows=2 + 5)


def test_non_size_fit_failure_skips_the_search_and_fails_open(monkeypatch):
    w = _relayout_windows()
    meter = ForcedRelayoutMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([w[n] for n in "AB"]), runtime, meter=meter)
    meter.calls = 0
    meter.full_counts.clear()
    inputs = {"token_ceiling": 10_000, "byte_ceiling": 1_000_000, "source_plan": {},
              "sources": {}, "receipt": None, "measurement_identity": "other-tokenizer"}
    body = _body([w[n] for n in "ABC"])
    projected, _, journal = _invoke(monkeypatch, body, runtime, meter=meter, inputs=inputs)
    assert projected == body
    assert meter.calls == 1  # one measurement, no relayout search
    failed = [e for e in journal.events if e["kind"] == "stable_wire_failed_open"]
    assert len(failed) == 1 and failed[0]["cause"] == "measurement_method_mismatch"
    assert not _epochs(journal)


def test_rerendered_window_caps_the_kept_prefix(monkeypatch):
    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([w[n] for n in "ABCD"]), runtime)
    rerendered = copy.deepcopy(w["C"])
    rerendered["content"] = rerendered["content"].replace('"max_lines":1000', '"max_lines":999')
    projected, _, journal = _invoke(
        monkeypatch, _body([w["A"], w["B"], rerendered, w["D"]]), runtime)
    assert _sources(projected) == [w["A"], w["B"], rerendered, w["D"]]
    event = _epochs(journal)[0]
    assert event["reason"] == "rerender" and event["kept_prefix_windows"] == 2


def test_structural_relayout_still_does_a_full_first_seen_sort(monkeypatch):
    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _late_z_layout(monkeypatch, w, runtime)
    changed = _body([w["A"], w["B"], w["C"], w["D"], w["Z"]])
    changed["messages"][1]["content"] = "Task: yet another structure"
    projected, _, journal = _invoke(monkeypatch, changed, runtime)
    assert _sources(projected) == [w[n] for n in "ZABCD"]
    event = _epochs(journal)[0]
    assert event["reason"] == "structural" and event["kept_prefix_windows"] == 0


def test_appended_rows_follow_persisted_first_seen_order_across_three_sends(monkeypatch):
    w = _relayout_windows()
    meter = ForcedRelayoutMeter(max_sources=4)
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([w["D"], w["B"]]), runtime, meter=meter)

    meter.reject_next = True
    second, _, _ = _invoke(monkeypatch, _body([w["C"], w["A"], w["B"], w["D"]]),
                           runtime, meter=meter)
    # First seen: D, B, then C, A. The kept prefix is D,B and the rest follows first-seen order.
    assert _sources(second) == [w["D"], w["B"], w["C"], w["A"]]

    meter.reject_next = True
    third, _, journal = _invoke(monkeypatch, _body([w["F"], w["E"], w["A"], w["C"]]),
                                runtime, meter=meter)
    # Only k=0 fits four windows, so the sort is persisted first-seen (C, A), then F, E.
    assert _sources(third) == [w["C"], w["A"], w["F"], w["E"]]
    assert _epochs(journal)[0]["kept_prefix_windows"] == 0


def test_retain_prefix_switch_defaults_to_off_and_rejects_bad_values():
    from kernel_stable_wire import RETAIN_PREFIX_ENV, retain_prefix

    assert RETAIN_PREFIX_ENV == "STATEFRAME_STABLE_WIRE_RETAIN_PREFIX"
    assert retain_prefix({}) is False
    assert retain_prefix({RETAIN_PREFIX_ENV: " ON "}) is True
    assert retain_prefix({RETAIN_PREFIX_ENV: "off"}) is False
    with pytest.raises(ValueError, match="invalid"):
        retain_prefix({RETAIN_PREFIX_ENV: "maybe"})


@pytest.mark.parametrize("retain, leaver, expected, kept", [
    # ON keeps the not-current first window in the prefix; OFF stops at it (k=0).
    ("on", "A", "ABCDZE", 4),
    ("off", "A", "ZBCDE", 0),
    # A window leaving at the end never blocks the prefix in either mode.
    ("on", "D", "ABCDZE", 4),
    ("off", "D", "ABCZE", 3),
])
def test_relayout_prefix_by_retain_mode_keeps_exact_bytes(monkeypatch, retain, leaver, expected, kept):
    w = _relayout_windows()
    meter = ForcedRelayoutMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([w[n] for n in "ZABCD"]), runtime, meter=meter, retain=retain)
    first, _, _ = _invoke(monkeypatch, _relayout_body([w[n] for n in "ABCD"]), runtime,
                          meter=meter, retain=retain)
    meter.reject_next = True
    current = [w[n] for n in "ABCD" if n != leaver] + [w["Z"], w["E"]]
    second, _, journal = _invoke(monkeypatch, _relayout_body(current[::-1]), runtime,
                                 meter=meter, retain=retain)
    assert _sources(second) == [w[n] for n in expected]
    event = _epochs(journal)[0]
    assert event["kept_prefix_windows"] == kept and event["retain_prefix"] is (retain == "on")
    head = 2 + kept
    _assert_stable_message_byte_prefix(first, second, rows=head)
    assert _fake_tokens(second["messages"][:head]) == _fake_tokens(first["messages"][:head])


def test_retain_off_epoch_never_keeps_a_not_current_row_in_the_prefix(monkeypatch):
    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([w[n] for n in "ABC"]), runtime, retain="off")
    rerendered = copy.deepcopy(w["C"])
    rerendered["content"] = rerendered["content"].replace('"max_lines":1000', '"max_lines":999')
    # B left and C re-rendered: the rerender relayout keeps A only, no retained rows.
    projected, _, journal = _invoke(monkeypatch, _body([w["A"], rerendered]), runtime,
                                    retain="off")
    assert _sources(projected) == [w["A"], rerendered]
    assert "Not current, do not edit from these:" not in _status_rows(projected["messages"])[0]["content"]
    assert _epochs(journal)[0]["kept_prefix_windows"] == 1


def _recorded_k2_sends(monkeypatch):
    """Record an envelope-forced k=2 relayout and return the sends' bodies and results."""
    w = _relayout_windows()
    meter = ForcedRelayoutMeter(max_sources=4)
    runtime = SimpleNamespace(_stable_wire_state=None)
    bodies = [_body([w[n] for n in "ABCD"]),
              _body([w["F"], w["E"], w["A"], w["B"]])]
    results = []
    meter.reject_next = False
    for index, body in enumerate(bodies):
        meter.reject_next = index == 1
        result, _, journal = _invoke(monkeypatch, copy.deepcopy(body), runtime, meter=meter)
        results.append((result, journal))
    assert _epochs(results[1][1])[0]["kept_prefix_windows"] == 2
    return w, bodies, results


def test_replay_reproduces_the_recorded_kept_prefix(monkeypatch):
    import kernel_cache_layout_candidate as candidate
    from kernel_stable_wire import recorded_window_layout

    _w, bodies, recorded = _recorded_k2_sends(monkeypatch)
    queue = [recorded_window_layout(result["messages"]) for result, _ in recorded]
    cursor = {"send": 0}
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER",
                        lambda: queue[cursor["send"]])
    # The replay meter never rejects: the recorded request alone decides the kept prefix.
    meter = ForcedRelayoutMeter(max_sources=99)
    runtime = SimpleNamespace(_stable_wire_state=None)
    for index, body in enumerate(bodies):
        cursor["send"] = index
        replayed, _, journal = _invoke(monkeypatch, copy.deepcopy(body), runtime, meter=meter)
        assert replayed == recorded[index][0]
        assert (json.dumps(replayed, sort_keys=True).encode()
                == json.dumps(recorded[index][0], sort_keys=True).encode())
    assert _epochs(journal)[0]["reason"] == "envelope"
    assert _epochs(journal)[0]["kept_prefix_windows"] == 2
    # No replay fit decisions. These sends had zero recorded trims; final bytes match above.
    assert meter.full_counts == []
    assert not _events(journal, "stable_wire_replay_mismatch")
    assert not _events(journal, "stable_wire_failed_open")


def test_replay_applied_without_windows_provider_fails_open(monkeypatch):
    import kernel_cache_layout_candidate as candidate

    _w, bodies, _recorded = _recorded_k2_sends(monkeypatch)
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", None)
    runtime = SimpleNamespace(_stable_wire_state=None)
    for body in bodies:
        plain = copy.deepcopy(body)
        replayed, _, journal = _invoke(monkeypatch, plain, runtime,
                                 meter=ForcedRelayoutMeter(max_sources=99))
        assert replayed is plain
        assert len(_events(journal, "stable_wire_replay_mismatch")) == 1
        assert _events(journal, "stable_wire_failed_open")[0]["reason"] == "replay_layout_mismatch"
    assert runtime._stable_wire_state is None


def test_replay_relayout_with_all_windows_uses_recorded_status_epoch(monkeypatch):
    from kernel_stable_wire import recorded_window_layout

    w = _relayout_windows()
    meter = ForcedRelayoutMeter()
    runtime = SimpleNamespace(_stable_wire_state=None)
    bodies, recorded = [], []
    # Reach epoch 9 with real structural sends. At 9 -> 10 the status size changes,
    # so the relayout chooser probes k=all rather than skipping an identical envelope.
    for index in range(9):
        body = _body([w["A"], w["B"]])
        body["messages"][1]["content"] = f"Task: epoch case {index}"
        result, _, _ = _invoke(monkeypatch, body, runtime, meter=meter)
        bodies.append(body)
        recorded.append(result)
    first = recorded[-1]
    meter.reject_next = True
    # Append-only is rejected on size, but k=all fits on the next probe, keeping the windows.
    second, _, journal = _invoke(monkeypatch, bodies[-1], runtime, meter=meter)
    bodies.append(copy.deepcopy(bodies[-1]))
    recorded.append(second)
    assert _epochs(journal)[0]["reason"] == "envelope"
    assert _epochs(journal)[0]["kept_prefix_windows"] == 2
    assert _sources(second) == _sources(first)
    assert recorded_window_layout(second["messages"])[1] == 10

    queue = [recorded_window_layout(value["messages"]) for value in recorded]
    cursor = {"send": 0}
    _replay_with(monkeypatch, lambda: queue[cursor["send"]])
    runtime = SimpleNamespace(_stable_wire_state=None)
    for index, (body, expected) in enumerate(zip(bodies, recorded)):
        cursor["send"] = index
        replayed, _, journal = _invoke(monkeypatch, body, runtime)
        assert json.dumps(replayed).encode() == json.dumps(expected).encode()
        assert not _events(journal, "stable_wire_replay_mismatch")

    # Identical final windows cannot hide an unprovable status epoch.
    queue[-1] = (queue[-1][0], 99)
    result, _, journal = _invoke(monkeypatch, bodies[-1], runtime)
    assert not any(row.get("name") == "stateframe_stable_wire_status" for row in result["messages"])
    mismatch, = _events(journal, "stable_wire_replay_mismatch")
    assert mismatch["recorded_epoch"] == 99
    assert mismatch["shared_prefix_windows"] == 2
    assert _events(journal, "stable_wire_failed_open")[0]["reason"] == "replay_layout_mismatch"
    assert runtime._stable_wire_state["epoch"] == 10


def _events(journal, kind):
    return [event for event in journal.events if event["kind"] == kind]


def _replay_with(monkeypatch, provider):
    import kernel_cache_layout_candidate as candidate

    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_DECISION_PROVIDER", lambda: "applied")
    monkeypatch.setattr(candidate, "_REPLAY_STABLE_WIRE_WINDOWS_PROVIDER", provider)


def _assert_replay_failed_open(result, body, journal, *, reason, recorded, relayout):
    """The resumed send carried the plain body, named the reason and leaked no content."""
    assert result is body
    mismatch = _events(journal, "stable_wire_replay_mismatch")
    failed = _events(journal, "stable_wire_failed_open")
    assert len(mismatch) == 1 and len(failed) == 1
    assert failed[0]["reason"] == "replay_layout_mismatch"
    for event in (mismatch[0], failed[0]):
        assert event["recorded_window_count"] == len(recorded)
        assert event["relayout"] is relayout
        assert all(not isinstance(value, (list, dict)) for value in event.values())
        assert not any("[Current source:" in str(value) for value in event.values())
    assert mismatch[0]["layout_reason"] == failed[0]["layout_reason"] == reason
    assert mismatch[0]["shared_prefix_windows"] < len(recorded)


def test_replay_of_an_unreproducible_recorded_prefix_fails_open_with_an_event(monkeypatch):
    from kernel_stable_wire import recorded_window_layout

    w, bodies, recorded = _recorded_k2_sends(monkeypatch)
    queue = [recorded_window_layout(result["messages"]) for result, _ in recorded]
    # No kept prefix reproduces this order: F ahead of A,B is impossible from a first-seen sort.
    windows, epoch = queue[1]
    queue[1] = ([windows[2], windows[3], windows[0], windows[1]], epoch)
    assert queue[1][0] != [str(w[n]["content"]) for n in "ABFE"]
    cursor = {"send": 0}
    _replay_with(monkeypatch, lambda: queue[cursor["send"]])
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, copy.deepcopy(bodies[0]), runtime, meter=ForcedRelayoutMeter(max_sources=99))
    cursor["send"] = 1
    body = copy.deepcopy(bodies[1])
    result, _, journal = _invoke(monkeypatch, body, runtime, meter=ForcedRelayoutMeter(max_sources=99))
    _assert_replay_failed_open(result, body, journal, reason="envelope", recorded=queue[1][0],
                               relayout=True)
    # The failed-open send never became delivered state for the next send.
    assert runtime._stable_wire_state["epoch"] == 1


def test_replay_append_only_mismatch_fails_open_instead_of_a_silent_relayout(monkeypatch):
    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    # Recording says the second send relayouted to an order no kept prefix reproduces.
    queue = [([str(w[n]["content"]) for n in "AB"], 1),
             ([str(w[n]["content"]) for n in "CBA"], 2)]
    cursor = {"send": 0}
    _replay_with(monkeypatch, lambda: queue[cursor["send"]])
    _invoke(monkeypatch, _body([w["A"], w["B"]]), runtime)
    cursor["send"] = 1
    body = _body([w["A"], w["B"], w["C"]])
    result, _, journal = _invoke(monkeypatch, body, runtime)
    _assert_replay_failed_open(result, body, journal, reason="envelope", recorded=queue[1][0],
                               relayout=True)


def _structural_sends(monkeypatch, runtime, *, provider=None):
    """First send A,B; second send changes the task text so the windows relayout structurally."""
    w = _relayout_windows()
    if provider is not None:
        _replay_with(monkeypatch, provider)
    first = _invoke(monkeypatch, _body([w["A"], w["B"]]), runtime)
    rebased = _body([w["C"], w["B"], w["A"]])
    rebased["messages"][1]["content"] = "Task: a structurally new task"
    return w, first, rebased


def test_replay_structural_relayout_checks_the_recorded_rows(monkeypatch):
    w = _relayout_windows()
    cursor = {"send": 0}
    # The recording kept C ahead of A: a structural first-seen sort can never reproduce it.
    queue = [([str(w[n]["content"]) for n in "AB"], 1),
             ([str(w[n]["content"]) for n in "CAB"], 2)]
    runtime = SimpleNamespace(_stable_wire_state=None)
    _, _first, rebased = _structural_sends(monkeypatch, runtime,
                                           provider=lambda: queue[cursor["send"]])
    cursor["send"] = 1
    result, _, journal = _invoke(monkeypatch, rebased, runtime)
    assert _epochs(journal) == []  # nothing stable was sent, so no epoch was announced
    _assert_replay_failed_open(result, rebased, journal, reason="structural",
                               recorded=queue[1][0], relayout=True)


def test_replay_structural_relayout_with_a_matching_recording_is_byte_exact(monkeypatch):
    from kernel_stable_wire import recorded_window_layout

    runtime = SimpleNamespace(_stable_wire_state=None)
    _w, first, rebased = _structural_sends(monkeypatch, runtime)
    recorded_second, _, live_journal = _invoke(monkeypatch, copy.deepcopy(rebased), runtime)
    assert _epochs(live_journal)[0]["reason"] == "structural"
    queue = [recorded_window_layout(first[0]["messages"]),
             recorded_window_layout(recorded_second["messages"])]
    cursor = {"send": 0}
    replay_runtime = SimpleNamespace(_stable_wire_state=None)
    _, _, replayed_rebased = _structural_sends(monkeypatch, replay_runtime,
                                               provider=lambda: queue[cursor["send"]])
    cursor["send"] = 1
    replayed, _, journal = _invoke(monkeypatch, replayed_rebased, replay_runtime)
    assert (json.dumps(replayed, sort_keys=True).encode()
            == json.dumps(recorded_second, sort_keys=True).encode())
    assert _epochs(journal)[0]["reason"] == "structural"
    assert not _events(journal, "stable_wire_replay_mismatch")
    assert not _events(journal, "stable_wire_failed_open")
