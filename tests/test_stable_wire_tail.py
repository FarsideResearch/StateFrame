"""Stable wire keeps every per-send block in the volatile tail, behind the source windows."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from kernel_request_source_window import (
    DEFERRED_SOURCE_NOTICE_HEADER,
    _render_truthful_deferred_source_notice,
)
from kernel_stable_wire import canonicalize_stable_wire
from test_stable_wire import (
    Journal,
    Meter,
    _body,
    _final_layout_body,
    _invoke,
    _render_final_layout_stable,
    _revision,
    _window,
)

DIET_HEADER = "[Request diet: dropped source ranges]"
WINDOW_PREFIX = "[Current source:"


def _notice(*delivered, paths=("pkg/z.py",)):
    return {"role": "user", "content": _render_truthful_deferred_source_notice(
        paths, delivered_source_ranges=list(delivered))}


def _host(text):
    return {"role": "user", "content": f"[Host fact] {text}"}


def _diet(text):
    return {"role": "user", "content": f"{DIET_HEADER}\n{text}"}


def _send(windows, per_send, history=()):
    """One pre-projection request: head rows, per-send rows ahead of windows, then the frame."""
    body = _body(windows)
    rows = body["messages"]
    rows[2:2] = copy.deepcopy(list(per_send))
    rows[-1:-1] = copy.deepcopy(list(history))
    return body


def _header(row):
    text = str(row.get("content", ""))
    return text.split("\n", 1)[0][:80]


def _head_rows(messages):
    """Rows of the cacheable head: system, task and source windows, ahead of the frame."""
    stop = next(index for index, row in enumerate(messages)
                if str(row.get("content", "")).startswith("[StateFrame]\n"))
    return messages[:stop]


def _assert_head_is_append_only(previous, current, label):
    before, after = _head_rows(previous["messages"]), _head_rows(current["messages"])
    for index, row in enumerate(before):
        if index >= len(after) or after[index] != row:
            offender = after[index] if index < len(after) else {}
            raise AssertionError(
                f"{label}: first differing message is index {index}, header "
                f"{_header(offender)!r} (was {_header(row)!r})")


def _assert_per_send_rows_behind_windows(messages, label):
    last_window = max(i for i, row in enumerate(messages)
                      if str(row.get("content", "")).startswith(WINDOW_PREFIX))
    first_window = min(i for i, row in enumerate(messages)
                       if str(row.get("content", "")).startswith(WINDOW_PREFIX))
    for index, row in enumerate(messages):
        content = str(row.get("content", ""))
        if content.startswith((DEFERRED_SOURCE_NOTICE_HEADER, "[Host fact]", DIET_HEADER)):
            assert index > last_window, (
                f"{label}: per-send section {_header(row)!r} sits at message index {index}, "
                f"ahead of the source windows ({first_window}..{last_window})")


def _scripted_sends():
    a = _window(path="pkg/a.py", revision=_revision("1"))
    b = _window(path="pkg/b.py", revision=_revision("2"))
    c = _window(path="pkg/c.py", revision=_revision("3"))
    d = _window(path="pkg/d.py", revision=_revision("4"))
    return [
        _send([a], [_notice(("pkg/a.py", _revision("1"), 1, 1))], []),
        _send([a, b], [_host("pkg/b.py lines 2-9 omitted; read loads them.")], []),
        _send([a, b, c], [_notice(("pkg/a.py", _revision("1"), 1, 1),
                                  ("pkg/b.py", _revision("2"), 1, 1), paths=("pkg/y.py", "pkg/z.py")),
                          _host("pkg/c.py lines 4-8 omitted; read loads them."),
                          _diet("dropped pkg/a.py:1-1 for this request")], []),
        _send([a, b, c, d], [], []),
        _send([a, b, c, d], [_notice(("pkg/d.py", _revision("4"), 1, 1), paths=("pkg/q.py",)),
                             _host("pkg/q.py is not attached; read loads it")], []),
    ]


def test_per_send_notice_and_host_facts_never_break_the_stable_head(monkeypatch):
    runtime = SimpleNamespace(_stable_wire_state=None)
    projected = []
    for number, body in enumerate(_scripted_sends(), start=1):
        result, _capture, _journal = _invoke(monkeypatch, body, runtime)
        _assert_per_send_rows_behind_windows(result["messages"], f"send {number}")
        projected.append(result)
    for number, (previous, current) in enumerate(zip(projected, projected[1:]), start=1):
        _assert_head_is_append_only(previous, current, f"send {number} -> {number + 1}")
    # Nothing per-send may precede the first window, so the serialized bytes ahead of it match.
    heads = []
    for result in projected:
        rows = result["messages"]
        first = next(i for i, row in enumerate(rows)
                     if str(row.get("content", "")).startswith(WINDOW_PREFIX))
        heads.append(json.dumps(rows[:first], sort_keys=True))
    assert len(set(heads)) == 1, "bytes before the first source window changed between sends"


def test_notice_and_host_facts_keep_their_text_and_recorded_order_in_the_tail(monkeypatch):
    notice = _notice(("pkg/a.py", _revision("1"), 1, 1), paths=("pkg/z.py",))
    host = _host("pkg/a.py lines 2-4 omitted; read loads them.")
    diet = _diet("dropped pkg/a.py:1-1 for this request")
    body = _send([_window(path="pkg/a.py", revision=_revision("1"))], [notice, diet, host])
    result, _capture, _journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    contents = [row["content"] for row in result["messages"]]
    for row in (notice, diet, host):
        assert contents.count(row["content"]) == 1
    tail = [c for c in contents if c in {notice["content"], diet["content"], host["content"]}]
    assert tail == [notice["content"], diet["content"], host["content"]]
    status = next(i for i, c in enumerate(contents) if c.startswith("[Source windows status]"))
    assert contents.index(notice["content"]) > status


def test_canonical_view_restores_the_notice_ahead_of_the_windows(monkeypatch):
    notice = _notice(("pkg/a.py", _revision("1"), 1, 1))
    body = _send([_window(path="pkg/a.py", revision=_revision("1"))], [notice])
    original = [row["content"] for row in body["messages"]]
    result, _capture, _journal = _invoke(monkeypatch, body, SimpleNamespace(_stable_wire_state=None))
    canonical, _index = canonicalize_stable_wire(result["messages"])
    canonical_contents = [row["content"] for row in canonical
                          if not str(row["content"]).startswith("[Source windows status]")]
    assert canonical_contents.index(notice["content"]) < next(
        i for i, c in enumerate(canonical_contents) if c.startswith(WINDOW_PREFIX))
    assert canonical_contents[:3] == original[:3]


def test_final_layout_driver_keeps_head_stable_across_changing_per_send_rows(monkeypatch):
    runtime = SimpleNamespace(_stable_wire_state=None)
    windows = [_window(path=f"pkg/{n}.py", revision=_revision(str(i)))
               for i, n in enumerate("abc", start=1)]
    per_send = [[_notice(("pkg/a.py", _revision("1"), 1, 1))],
                [_host("pkg/b.py lines 2-9 omitted; read loads them.")],
                [_notice(("pkg/b.py", _revision("2"), 1, 1), paths=("pkg/y.py",)),
                 _host("pkg/c.py lines 4-8 omitted; read loads them.")]]
    results = []
    for count, rows in enumerate(per_send, start=1):
        template = _final_layout_body()
        frame = next(r for r in template["messages"]
                     if str(r["content"]).startswith("[StateFrame]\n"))
        template["messages"] = [
            template["messages"][0], template["messages"][1], *copy.deepcopy(rows),
            *copy.deepcopy(windows[:count]), frame, template["messages"][-1]]
        _body_in, result, _capture, _rt, _inputs = _render_final_layout_stable(
            monkeypatch, template, runtime=runtime)
        _assert_per_send_rows_behind_windows(result["messages"], f"layout send {count}")
        results.append(result)
    for number, (previous, current) in enumerate(zip(results, results[1:]), start=1):
        _assert_head_is_append_only(previous, current, f"layout send {number} -> {number + 1}")


@pytest.mark.parametrize("flag", [None, "off"])
def test_off_leaves_per_send_rows_where_the_base_put_them(monkeypatch, flag):
    from kernel_cache_layout_candidate import render_final_cache_candidate

    if flag is None:
        monkeypatch.delenv("STATEFRAME_STABLE_WIRE", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_STABLE_WIRE", flag)
    monkeypatch.delenv("STATEFRAME_STABLE_WIRE_RETAIN_PREFIX", raising=False)
    monkeypatch.delenv("STATEFRAME_FINAL_CACHE_LAYOUT", raising=False)
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_a, **_k: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied", lambda _r: True)
    template = _final_layout_body()
    template["messages"][2:2] = [
        _notice(("pkg/a.py", _revision("1"), 1, 1)), _host("pkg/a.py lines 2-4 omitted; read loads them.")]
    inputs = {"token_ceiling": 10_000, "byte_ceiling": 1_000_000,
              "source_plan": {}, "sources": {}, "receipt": None}
    result = render_final_cache_candidate(
        SimpleNamespace(journal=Journal(), request_source_final_delivery=None), None,
        copy.deepcopy(template), meter=Meter(), inputs=inputs)
    wire = json.dumps(result, ensure_ascii=False, sort_keys=True).encode("utf-8")
    # Verified against pristine v0.5.17 (9f07117d), with both stable-wire switches unset.
    assert hashlib.sha256(wire).hexdigest() == OFF_MODE_BASE_SHA256
    assert result["messages"][2]["content"].startswith(DEFERRED_SOURCE_NOTICE_HEADER)


OFF_MODE_BASE_SHA256 = "d1cfb962488cc0c16b5689f54d9198b5b43bd052be2d225a7087364e1d4d0543"


def test_deferred_notice_is_still_found_and_verified_in_its_tail_position(monkeypatch):
    from kernel_request_source_window import (
        _wire_deferred_notice_state,
        deferred_source_notice_paths,
    )

    notice = _notice(("pkg/a.py", _revision("1"), 1, 1), paths=("pkg/z.py",))
    template = _final_layout_body()
    template["messages"][2:2] = [copy.deepcopy(notice)]
    _in, result, _capture, _rt, _inputs = _render_final_layout_stable(monkeypatch, template)
    contents = [row["content"] for row in result["messages"]]
    assert notice["content"] in contents
    assert contents.index(notice["content"]) > next(
        i for i, c in enumerate(contents) if c.startswith(WINDOW_PREFIX))
    assert deferred_source_notice_paths(result["messages"]) == (("pkg/z.py",),)
    wire = {"messages": result["messages"], "tools": [{"function": {"name": "read"}}]}
    assert _wire_deferred_notice_state(wire, ("pkg/z.py",)) == (True, True)


def _section_name(row):
    content = str(row.get("content", ""))
    if row.get("role") == "system":
        return "system"
    if content.startswith(WINDOW_PREFIX):
        return f"window {content.split(chr(10), 1)[0][len(WINDOW_PREFIX):].strip(' ]')}"
    if content.startswith("[StateFrame]\n"):
        return "frame"
    return "task" if content.startswith("Task:") else _header(row)


def _through_last_window(messages):
    last = max(i for i, row in enumerate(messages)
               if str(row.get("content", "")).startswith(WINDOW_PREFIX))
    return messages[:last + 1]


def _assert_wire_prefix_through_last_window(previous, current, label):
    """Bytes from the start through the last window of `previous` prefix `current`'s wire."""
    before = _through_last_window(previous["messages"])
    after = current["messages"]
    for index, row in enumerate(before):
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        other = json.dumps(after[index], ensure_ascii=False, separators=(",", ":")) \
            if index < len(after) else ""
        if encoded != other:
            raise AssertionError(
                f"{label}: first differing section is {_section_name(row)!r} at message "
                f"index {index} (now {_section_name(after[index]) if index < len(after) else 'missing'!r})")


def test_frame_changing_every_send_never_relayouts_the_delivered_windows(monkeypatch):
    """Class killer: a frame change sits after the windows, so it cannot move window bytes.

    Each send names only its newest two windows as current, so a relayout would drop the
    older delivered windows from the wire; the append-only list must keep them in place.
    """
    runtime = SimpleNamespace(_stable_wire_state=None)
    windows = [_window(path=f"pkg/{n}.py", revision=_revision(str(i)))
               for i, n in enumerate("abcdef", start=1)]
    results, journals = [], []
    for count in range(1, len(windows) + 1):
        template = _final_layout_body()
        frame_row = next(r for r in template["messages"]
                         if str(r["content"]).startswith("[StateFrame]\n"))
        payload = json.loads(frame_row["content"].split("\n", 1)[1])
        payload["state"]["framefit5"]["stable_marker"] = f"send-{count}"
        frame_row["content"] = "[StateFrame]\n" + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        template["messages"] = [template["messages"][0], template["messages"][1],
                                *copy.deepcopy(windows[max(0, count - 2):count]), frame_row,
                                template["messages"][-1]]
        _in, result, capture, _rt, _inputs = _render_final_layout_stable(
            monkeypatch, template, runtime=runtime)
        results.append(result)
        journals.append(capture.journal)
    frames = {next(r["content"] for r in res["messages"]
                   if str(r.get("content", "")).startswith("[StateFrame]\n"))
              for res in results}
    assert len(frames) == len(results), "the stable frame payload must change on every send"
    for number, (previous, current) in enumerate(zip(results, results[1:]), start=1):
        _assert_wire_prefix_through_last_window(previous, current, f"send {number} -> {number + 1}")
    structural = [event for journal in journals for event in journal.events
                  if event["kind"] == "stable_wire_epoch" and event.get("reason") == "structural"]
    assert structural == []


def _replay_relayout(monkeypatch, *, trigger):
    """Replay sends through the real project_stable_wire; return (previous, relayout) wires."""
    from test_stable_wire import (SourceMeter, _late_z_layout, _relayout_body,
                                  _relayout_windows)

    w = _relayout_windows()
    runtime = SimpleNamespace(_stable_wire_state=None)
    costs = {row["content"]: 500 for row in w.values()}
    costs[w["C"]["content"]] = 7_000
    meter = SourceMeter(costs)
    # Previous layout is [a, b, c, d, z] while z is first in first-seen order.
    _late_z_layout(monkeypatch, w, runtime, headroom=6_000, meter=meter)
    previous = runtime._stable_wire_state["delivered"]
    previous_wire, _, _ = _invoke(monkeypatch, _relayout_body([w[n] for n in "ABCDZ"]),
                                  runtime, headroom=6_000, meter=meter)
    assert [r["identity"]["path"] for r in previous] == [f"pkg/{n}.py" for n in "ABCDZ"]
    if trigger == "rerender":
        # Mid window c is re-rendered (metadata only); a and b stay unchanged ahead of it.
        changed = copy.deepcopy(w["C"])
        changed["content"] = changed["content"].replace('"max_lines":1000', '"max_lines":999')
        meter = SourceMeter({**costs, changed["content"]: 500})
        current = [w["F"], w["E"], w["D"], w["Z"], changed, w["B"], w["A"]]
    else:
        current = [w["F"], w["E"], w["D"], w["Z"], w["B"], w["A"]]  # c dropped: 7000 retained
    relayout_wire, _, journal = _invoke(monkeypatch, _relayout_body(current), runtime,
                                        headroom=6_000, meter=meter)
    reasons = [e.get("reason") for e in journal.events if e["kind"] == "stable_wire_epoch"]
    assert reasons == [trigger], reasons
    return previous_wire, relayout_wire


@pytest.mark.parametrize("trigger", ("rerender", "headroom"))
def test_relayout_keeps_windows_before_the_first_changed_one_byte_identical(monkeypatch, trigger):
    """Class killer: a relayout may only change bytes from the first changed window onward.

    The previous layout has a window that was first seen earliest but came back last; a
    plain first-seen sort would put it first and move the first changed byte to window 1.
    """
    previous, relayout = _replay_relayout(monkeypatch, trigger=trigger)
    # a and b lead the previous layout and are unchanged, so the wire through them must match.
    keep = 2 + 2  # system + task rows, then windows a and b
    for index, row in enumerate(previous["messages"][:keep]):
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        other = json.dumps(relayout["messages"][index], ensure_ascii=False, separators=(",", ":"))
        if encoded != other:
            raise AssertionError(
                f"{trigger}: first differing section is {_section_name(row)!r} at message "
                f"index {index} (now {_section_name(relayout['messages'][index])!r})")


def test_send_whose_fresh_layout_is_just_over_the_envelope_stays_stable(monkeypatch):
    """Class killer: a layout a few tokens over the envelope is trimmed, never sent plain.

    Replays four sends through the real renderer. Send 3 has an envelope a hair under its
    fresh layout and a trim that lands short on its first ask, as the live run's did. The
    send must keep the stable layout and leave the delivered window prefix byte-identical.
    """
    from test_stable_wire_fit import SizeMeter, UnderTrimmer, _inputs, _layout_body, _render

    windows = [_window(path=f"pkg/{n}.py", revision=_revision(str(i)))
               for i, n in enumerate("abcd", start=1)]
    runtime = SimpleNamespace(_stable_wire_state=None)
    results = []
    for count in range(1, 5):
        def send_body():
            body = _layout_body(optional_rows=6)
            rows = body["messages"]
            at = next(i for i, r in enumerate(rows) if str(r["content"]).startswith(WINDOW_PREFIX))
            rows[at:at + 1] = copy.deepcopy(windows[:count])
            return body

        ceiling = 100_000
        trimmer = None
        if count == 3:
            probe, _ = _render(monkeypatch, send_body(), meter=SizeMeter(), inputs=_inputs(100_000),
                               runtime=SimpleNamespace(_stable_wire_state=copy.deepcopy(
                                   runtime._stable_wire_state)))
            ceiling = SizeMeter().measure(probe)[0] - 12
        body = send_body()
        if count == 3:
            trimmer = UnderTrimmer(body, rows_per_reserve=0.05)
        result, capture = _render(monkeypatch, body, meter=SizeMeter(), inputs=_inputs(ceiling),
                                  trim=trimmer, runtime=runtime)
        failed = [e for e in capture.journal.events if e["kind"] == "stable_wire_failed_open"]
        assert not failed, f"send {count} fell back to the plain layout: {failed[0]}"
        assert capture._final_cache_layout_record["stable_wire"] == "applied", f"send {count}"
        assert SizeMeter().measure(result)[0] <= ceiling, f"send {count} exceeds its envelope"
        if count == 3:
            assert len(trimmer.calls) >= 2, "the trim was expected to land short once"
        results.append(result)
    for number, (previous, current) in enumerate(zip(results, results[1:]), start=1):
        _assert_wire_prefix_through_last_window(previous, current, f"send {number} -> {number + 1}")
