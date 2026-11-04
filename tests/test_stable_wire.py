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


def _invoke(monkeypatch, body, runtime, *, headroom=16_000, meter=None, inputs=None):
    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
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

    # Planted control: same path/revision/range with one changed byte must refuse.
    changed = copy.deepcopy(first)
    changed["content"] = changed["content"].replace("value = 1", "value = 9")
    control_runtime = SimpleNamespace(_stable_wire_state=copy.deepcopy(old_runtime._stable_wire_state))
    with pytest.raises(ValueError, match="immutable|re-render|stable wire"):
        _invoke(monkeypatch, _body([changed]), control_runtime)


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
    meter = SourceMeter({row["content"]: 4_000 for row in (v1, v2, v3, other)})
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

    # v1 and v2 are now both retained, so one headroom rebase keeps current v3 only.
    third, _, events3 = _invoke(monkeypatch, _body([v3]), runtime, headroom=6_000, meter=meter)
    assert _messages_with_prefix(third["messages"], "[Current source:") == [v3]
    epochs = [row for row in events3.events if row["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0].get("reason") == "headroom"
    assert "\nEpoch 2\n" in _status_rows(third["messages"])[0]["content"]

    # The compacted epoch retains only v3, and a later new window appends normally.
    fourth, _, events4 = _invoke(monkeypatch, _body([v3, other]), runtime,
                                 headroom=6_000, meter=meter)
    assert _messages_with_prefix(fourth["messages"], "[Current source:") == [v3, other]
    assert not any(row["kind"] == "stable_wire_epoch" for row in events4.events)


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


def test_stable_frame_change_rebases_before_same_epoch_message_prefix_changes(monkeypatch):
    old = _window(path="pkg/old.py", revision=_revision("1"))
    current = _window(path="pkg/current.py", revision=_revision("2"))
    runtime = SimpleNamespace(_stable_wire_state=None)
    _invoke(monkeypatch, _body([old]), runtime)

    payload = _payload()
    payload["state"]["framefit5"]["stable_marker"] = "changed"
    projected, _, journal = _invoke(monkeypatch, _body([current], payload=payload), runtime)
    assert _messages_with_prefix(projected["messages"], "[Current source:") == [current]
    epochs = [event for event in journal.events if event["kind"] == "stable_wire_epoch"]
    assert len(epochs) == 1 and epochs[0].get("reason") == "structural"


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
    from stateframe._child import install_wire_recorder
    monkeypatch.setattr(kernel_bounded_http, "request_bytes",
                        lambda *_args, **_kwargs: b"{}")
    run = tmp_path / "failed-open-send"
    run.mkdir()
    install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions")
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
    from stateframe._child import install_wire_recorder
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
    monkeypatch.setattr(kernel_bounded_http, "request_bytes", kernel_bounded_http.request_bytes)
    run = tmp_path / "replay-run"
    run.mkdir()
    install_wire_recorder(run, "http://127.0.0.1:8000/v1/chat/completions", replay)

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
