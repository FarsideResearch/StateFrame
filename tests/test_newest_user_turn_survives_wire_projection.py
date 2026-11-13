"""The newest user turn stays contiguous with the tool exchange through every wire projection.

A live run with STATEFRAME_STABLE_WIRE=on crashed with "newest user suffix changed during
allocation". In the final cache layout the frame sits just before the exchange, so the newest
user turn (here a host advisory) is physically before the frame. The stable-wire projection
treated every pre-frame row as cacheable prefix: it hoisted the advisory ahead of the source
windows, and the canonical view put the source-fit summary where the advisory belonged. The
projection now keeps every per-request row in recorded order after the status row, proves the
order or fails open to the plain wire, and allocation re-prepares the request on a suffix
change instead of ending the run.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

import kernel_stable_wire as stable_wire
from kernel_request_source_window import DEFERRED_SOURCE_NOTICE_HEADER
from kernel_stable_wire import canonicalize_stable_wire
from test_stable_wire import _body, _invoke, _window

SUMMARY = "kept 1 selected source windows; about 322 chars of source fit"
ADVISORY = "[kernel] You have taken ZERO actions in 50 steps of pure reading. Pick one."


def _final_cache_shaped_body():
    """Rows as the final cache layout hands them to the projection: frame before exchange."""
    body = _body([_window()])
    messages = body["messages"]
    frame_index = next(i for i, row in enumerate(messages)
                       if str(row.get("content", "")).startswith("[StateFrame]\n"))
    frame = messages[frame_index]
    tail = messages[frame_index + 1:]
    exchange = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call-1", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "[read page] path=pkg/a.py"},
    ]
    messages[:] = [
        messages[0], messages[1],
        {"role": "user", "content": "[Host fact] attached source bytes"},
        *messages[2:frame_index],
        {"role": "user", "content": SUMMARY},
        {"role": "user", "content": ADVISORY},
        frame, *exchange, *tail,
    ]
    return body


def _contents(rows):
    return [row.get("content") for row in rows]


def _before_exchange(rows):
    start = next(i for i, row in enumerate(rows) if row.get("role") == "assistant")
    return _contents(rows[:start])


def test_newest_user_turn_is_never_hoisted_into_the_cacheable_prefix(monkeypatch):
    body = _final_cache_shaped_body()
    result, _capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    rows = result["messages"]
    assert not [e for e in journal.events if e.get("kind") == "stable_wire_failed_open"]
    status = next(i for i, row in enumerate(rows)
                  if str(row.get("content", "")).startswith("[Source windows status]"))
    advisory = _contents(rows).index(ADVISORY)
    first_window = next(i for i, row in enumerate(rows)
                        if str(row.get("content", "")).startswith("[Current source:"))
    assert advisory > status > first_window
    # Physically, the newest user turn still directly precedes the exchange.
    assert _before_exchange(rows)[-2:] == [SUMMARY, ADVISORY]


def test_canonical_view_restores_the_recorded_order_before_the_frame(monkeypatch):
    body = _final_cache_shaped_body()
    original = copy.deepcopy(body["messages"])
    result, _capture, _journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    canonical, index_map = canonicalize_stable_wire(result["messages"])
    frame = next(i for i, row in enumerate(canonical)
                 if str(row.get("content", "")).startswith("[StateFrame]\n"))
    assert _contents(canonical[frame - 3:frame]) == [
        "[Host fact] attached source bytes", SUMMARY, ADVISORY]
    assert _contents(canonical[frame + 1:]) == _contents(
        original[[i for i, row in enumerate(original)
                  if str(row.get("content", "")).startswith("[StateFrame]\n")][0] + 1:])
    assert sorted(index_map.values()) == list(range(len(canonical)))


@pytest.mark.parametrize("newest", [
    [ADVISORY],
    ["[tool guidance] newest", ADVISORY],
    [ADVISORY, "[Host fact] newest"],
])
def test_any_mix_of_moved_and_plain_newest_rows_keeps_its_order(monkeypatch, newest):
    body = _final_cache_shaped_body()
    messages = body["messages"]
    at = _contents(messages).index(ADVISORY)
    messages[at:at + 1] = [{"role": "user", "content": text} for text in newest]
    result, _capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert not [e for e in journal.events if e.get("kind") == "stable_wire_failed_open"]
    canonical, _ = canonicalize_stable_wire(result["messages"])
    frame = next(i for i, row in enumerate(canonical)
                 if str(row.get("content", "")).startswith("[StateFrame]\n"))
    assert _contents(canonical[frame - len(newest):frame]) == newest


def test_a_projection_that_would_reorder_rows_fails_open_to_the_plain_wire(monkeypatch):
    real = stable_wire._project_body

    def swapping(body, delivered, current, epoch, *, ownership, prove=True):
        projected, status, frame = real(
            body, delivered, current, epoch, ownership=ownership, prove=prove)
        rows = projected["messages"]
        a, b = _contents(rows).index(SUMMARY), _contents(rows).index(ADVISORY)
        rows[a], rows[b] = rows[b], rows[a]
        return projected, status, frame

    monkeypatch.setattr(stable_wire, "_project_body", swapping)
    body = _final_cache_shaped_body()
    plain = copy.deepcopy(body)
    result, _capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert result == plain
    reasons = [e.get("reason") for e in journal.events if e.get("kind") == "stable_wire_failed_open"]
    assert reasons == ["projection_reorders_messages"]


def test_older_projections_without_late_rows_canonicalize_as_before(monkeypatch):
    # Projections recorded before this fix put moved rows, and only moved rows, after the
    # status row. Their canonical view must not change, or replayed checkpoints would drift.
    body = _body([_window()])
    messages = body["messages"]
    messages[2:2] = [{"role": "user", "content": "[Host fact] before-frame"},
                     {"role": "user", "content": SUMMARY}]
    result, _capture, _journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    canonical, _ = canonicalize_stable_wire(result["messages"])
    assert _contents(canonical)[:2] == ["system", "Task: keep source bytes stable"]
    frame = next(i for i, row in enumerate(canonical)
                 if str(row.get("content", "")).startswith("[StateFrame]\n"))
    assert _contents(canonical[frame - 2:frame]) == ["[Host fact] before-frame", SUMMARY]


def test_suffix_change_during_allocation_asks_for_one_replan_not_a_crash(monkeypatch):
    import kernel_frame_allocation_capture as allocation
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded

    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda _receipt: True)
    from kernel_stateframe_framefit5 import PRODUCT_MODE
    frame = "[StateFrame]\n" + json.dumps({"state": {"framefit5": {"mode": PRODUCT_MODE}}})
    exchange = [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    body = {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: x"},
                         {"role": "user", "content": frame},
                         {"role": "user", "content": SUMMARY},
                         *exchange]}
    capture = SimpleNamespace(_framefit5_newest_user_suffix=({"role": "user", "content": ADVISORY},))
    with pytest.raises(FrameAllocationReplanNeeded, match="newest user suffix changed"):
        allocation.capture_candidate(
            runtime=None, capture=capture, body=body, source_plan={}, receipt={},
            sources={}, optional_inputs=({}, {}), byte_ceiling=1, token_ceiling=1,
            measurement_identity="m")


def _allocation_body(suffix_rows):
    """A final-layout body whose newest user turn sits just before the exchange."""
    from kernel_stateframe_framefit5 import PRODUCT_MODE
    frame = "[StateFrame]\n" + json.dumps({"state": {"framefit5": {"mode": PRODUCT_MODE}}})
    exchange = [{"role": "assistant", "content": "", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    return {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: x"},
                         {"role": "user", "content": frame},
                         {"role": "user", "content": SUMMARY},
                         *suffix_rows,
                         *exchange]}


@pytest.mark.parametrize("mutate", ["edit", "insert", "move"], ids=["edited", "inserted", "moved"])
def test_newest_user_suffix_mutated_between_capture_and_allocation_never_ends_the_run(
        monkeypatch, mutate):
    # Class killer: whatever touches the newest user turn after capture (an edit, an
    # inserted row, or a row hoisted elsewhere) asks for one re-prepare. It is never
    # the bare FrameRequestAllocationError that ends a run.
    import kernel_frame_allocation_capture as allocation
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded

    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda _receipt: True)
    captured = ({"role": "user", "content": ADVISORY},)
    capture = SimpleNamespace(_framefit5_newest_user_suffix=captured)
    body = _allocation_body([copy.deepcopy(dict(row)) for row in captured])
    messages = body["messages"]
    at = _contents(messages).index(ADVISORY)
    if mutate == "edit":
        messages[at]["content"] = ADVISORY + " (edited)"
    elif mutate == "insert":
        messages.insert(at + 1, {"role": "user", "content": "[Host fact] late"})
    else:
        messages.insert(2, messages.pop(at))
    with pytest.raises(FrameAllocationReplanNeeded, match="newest user suffix changed"):
        allocation.capture_candidate(
            runtime=None, capture=capture, body=body, source_plan={}, receipt={},
            sources={}, optional_inputs=({}, {}), byte_ceiling=1, token_ceiling=1,
            measurement_identity="m")


def test_an_untouched_newest_user_suffix_is_not_a_replan(monkeypatch):
    # The guard above must fire only on a real change; an identical suffix passes it.
    import kernel_frame_allocation_capture as allocation
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded

    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda _receipt: True)
    captured = ({"role": "user", "content": ADVISORY},)
    capture = SimpleNamespace(_framefit5_newest_user_suffix=captured)
    body = _allocation_body([copy.deepcopy(dict(row)) for row in captured])
    try:
        allocation.capture_candidate(
            runtime=None, capture=capture, body=body, source_plan={}, receipt={},
            sources={}, optional_inputs=({}, {}), byte_ceiling=1, token_ceiling=1,
            measurement_identity="m")
    except FrameAllocationReplanNeeded as error:
        assert "newest user suffix changed" not in str(error)
    except Exception as error:  # noqa: BLE001 - any later guard is fine, this one is not
        assert "newest user suffix changed" not in str(error)


def test_pre_window_notice_rows_do_not_trip_the_order_check(monkeypatch):
    # The projection moves deferred-source and diet notices to the volatile tail and the
    # canonical view puts them back ahead of the windows. That known move is not a reorder:
    # the projection must still apply (not fail open) and _order_preserved must say so.
    body = _final_cache_shaped_body()
    messages = body["messages"]
    first_window = next(i for i, row in enumerate(messages)
                        if str(row.get("content", "")).startswith("[Current source:"))
    notice = DEFERRED_SOURCE_NOTICE_HEADER + "\npkg/b.py"
    diet = stable_wire.DIET_NOTICE_HEADER + "\npkg/a.py:40-80"
    messages.insert(first_window, {"role": "user", "content": notice})
    at = _contents(messages).index(SUMMARY)
    messages.insert(at, {"role": "user", "content": diet})
    original = copy.deepcopy(messages)
    result, _capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert not [e for e in journal.events if e.get("kind") == "stable_wire_failed_open"]
    assert stable_wire._order_preserved(original, result["messages"])
    rows = result["messages"]
    status = next(i for i, row in enumerate(rows)
                  if str(row.get("content", "")).startswith("[Source windows status]"))
    # Both notices follow the status row on the wire and lead the windows in the canonical view.
    assert _contents(rows).index(notice) > status and _contents(rows).index(diet) > status
    canonical, _ = canonicalize_stable_wire(rows)
    first_canonical_window = next(i for i, row in enumerate(canonical)
                                  if str(row.get("content", "")).startswith("[Current source:"))
    assert _contents(canonical).index(notice) < first_canonical_window
    assert _contents(canonical).index(diet) < first_canonical_window
    # The newest user turn still directly precedes the exchange.
    assert _before_exchange(rows)[-2:] == [SUMMARY, ADVISORY]


def test_order_check_rejects_a_swapped_newest_turn_without_a_notice_excuse():
    # Sanity for the skip above: a real reorder of plain rows is still caught.
    body = _final_cache_shaped_body()
    messages = body["messages"]
    projected = copy.deepcopy(messages)
    a, b = _contents(projected).index(SUMMARY), _contents(projected).index(ADVISORY)
    projected[a], projected[b] = projected[b], projected[a]
    assert not stable_wire._order_preserved(messages, projected)


NOTICE_CONTENTS = [
    DEFERRED_SOURCE_NOTICE_HEADER + "\npkg/b.py",
    stable_wire.DIET_NOTICE_HEADER + "\npkg/a.py:40-80",
    "[Host fact] native text",
    "[tool guidance] native text",
    "[Tool menu | this turn] native text",
    "[Source coverage] native text",
    SUMMARY,
]


@pytest.mark.parametrize("content", NOTICE_CONTENTS)
@pytest.mark.parametrize("metadata", [{"name": "read"}, {"tool_call_id": "call-1"}])
def test_native_user_metadata_never_identifies_a_moved_host_row(content, metadata):
    assert not stable_wire._is_moved_row({"role": "user", "content": content, **metadata})


@pytest.mark.parametrize("header", [DEFERRED_SOURCE_NOTICE_HEADER, stable_wire.DIET_NOTICE_HEADER])
def test_host_notice_header_must_be_a_complete_line(monkeypatch, header):
    body = _final_cache_shaped_body()
    lookalike = {"role": "user", "content": header + " is quoted here"}
    at = next(i for i, row in enumerate(body["messages"]) if row.get("role") == "assistant")
    body["messages"].insert(at - 1, lookalike)
    result, _capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert not [e for e in journal.events if e.get("kind") == "stable_wire_failed_open"]
    canonical, _ = canonicalize_stable_wire(result["messages"])
    frame = next(i for i, row in enumerate(canonical)
                 if str(row.get("content", "")).startswith("[StateFrame]\n"))
    assert canonical[frame - 1] == lookalike


@pytest.mark.parametrize("content", NOTICE_CONTENTS)
@pytest.mark.parametrize("kind", ["assistant", "tool", "user", "named_user", "tool_id_user"])
def test_notice_lookalikes_preserve_the_entire_native_exchange(monkeypatch, content, kind):
    body = _final_cache_shaped_body()
    rows = body["messages"]
    start = next(i for i, row in enumerate(rows) if row.get("role") == "assistant")
    if kind in {"assistant", "tool"}:
        rows[start + (kind == "tool")]["content"] = content
    else:
        row = {"role": "user", "content": content}
        if kind == "named_user":
            row["name"] = "native-user"
        elif kind == "tool_id_user":
            row["tool_call_id"] = "call-1"
        rows.insert(start + 2, row)
    exchange = copy.deepcopy(rows[start:])
    result, _capture, journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    assert not [e for e in journal.events if e.get("kind") == "stable_wire_failed_open"]
    for view in (result["messages"], canonicalize_stable_wire(result["messages"])[0]):
        at = next(i for i, row in enumerate(view) if row.get("role") == "assistant")
        assert view[at:] == exchange
    assert stable_wire._order_preserved(rows, result["messages"])


@pytest.mark.parametrize("content", NOTICE_CONTENTS[:2])
@pytest.mark.parametrize("role", ["assistant", "tool", "user"])
def test_order_check_cannot_skip_native_notice_lookalikes(content, role):
    messages = _final_cache_shaped_body()["messages"]
    if role == "user":
        messages.insert(-1, {"role": role, "content": content})
    else:
        next(row for row in messages if row.get("role") == role)["content"] = content
    reordered = copy.deepcopy(messages)
    at = next(i for i, row in enumerate(reordered)
              if row.get("role") == role and row.get("content") == content)
    reordered.append(reordered.pop(at))
    assert not stable_wire._order_preserved(messages, reordered)


def test_tool_deferred_notice_survives_real_final_source_verification(monkeypatch):
    from kernel_cache_layout_candidate import _checked_final_cache_projection
    from kernel_request_source_window import source_delivery_satisfied
    from test_stable_wire import Journal, Meter, _final_layout_body

    monkeypatch.setenv("STATEFRAME_STABLE_WIRE", "on")
    template = _final_layout_body()
    # No source obligations: exercise the real cache-layout validator and source
    # verifier without mocking either, using the supported new-output contract.
    template["messages"] = [row for row in template["messages"]
                            if not str(row.get("content", "")).startswith("[Current source:")]
    exchange = copy.deepcopy(_final_cache_shaped_body()["messages"][-3:-1])
    exchange[1]["content"] = DEFERRED_SOURCE_NOTICE_HEADER + "\npkg/b.py"
    template["messages"][-1:-1] = exchange
    body = _checked_final_cache_projection(template)
    original = copy.deepcopy(body)
    journal = Journal()
    capture = SimpleNamespace(journal=journal, request_id="native-notice", step=1,
                              request_source_final_delivery=None)
    result = stable_wire.project_stable_wire(
        capture, SimpleNamespace(_stable_wire_state=None), body, meter=Meter(),
        inputs={"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
                "source_plan": {"new_output_target": True, "workspace_authorization": {
                    "scope": "test-workspace", "generation": "1"}}, "sources": {}, "receipt": None})
    rows = result["messages"]
    at = next(i for i, row in enumerate(rows) if row.get("role") == "assistant")
    assert rows[at:at + 2] == exchange
    if stable_wire.is_stable_wire_projection(rows):
        assert source_delivery_satisfied(capture.request_source_final_delivery)
    else:
        assert result == original
        assert any(e.get("kind") == "stable_wire_failed_open" and e.get("reason")
                   for e in journal.events)
