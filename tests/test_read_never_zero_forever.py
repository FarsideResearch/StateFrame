"""A read that cannot fit must not stay zero forever.

Covers the 1-line progress bound, boundary probes priced without carrier equality,
the continuation carry across requests, and typed floor-infeasibility diagnosis.

Admission is driven directly through ``kernel_read_admission.admit``. The priced
envelope is a stub (``LINE_BYTES`` per admitted fresh line over a fixed ``BASE``);
the search, carry and journal logic under test is the real code.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernel_read_admission as admission
import kernel_working_set_live as live
from kernel_source_delivery import SourceDelivery, deliver_source, normalize_read_request
from kernel_working_set_stop import ReasonCode
from test_read_receipts_admission import Journal

BASE = 1000
LINE_BYTES = 100


def make_runtime(root, request_index=0):
    return SimpleNamespace(journal=Journal(root / "audit/events.jsonl"),
                           owner=SimpleNamespace(root=root), retained=None, step_ledger=None,
                           _frame_budget_request_count=request_index)


def write_lines(root, name, count, tag):
    (root / name).write_text("".join(f"{tag} {i:03d} " + "x" * 80 + "\n" for i in range(1, count + 1)))


def read(rt, root, name, args):
    raw = (root / name).read_bytes()
    request = normalize_read_request(args)
    source = deliver_source(raw, name, 0, args=request)
    return admission.admit(rt, raw=raw, display=name, start=0, request=request,
                           source=source, visible_messages=None)


def line_count(result):
    return len(result.content.splitlines())


def events(rt, **match):
    return [e for e in rt.journal.events
            if e["kind"] == admission.EVENT and all(e.get(k) == v for k, v in match.items())]


@pytest.fixture
def envelope(monkeypatch):
    """Stub priced envelope: BASE fixed bytes plus LINE_BYTES per proposed fresh line.

    ``knobs["allowance"]`` is the per-request fresh allowance. ``knobs["rule"]`` maps the
    proposed page line counts to extra violated limit names (empty means feasible).
    """
    knobs = {"allowance": 1000, "rule": lambda counts: ()}

    def snapshot(runtime, visible_messages, *, byte_cap):
        return SimpleNamespace(byte_cap=BASE + knobs["allowance"], base=BASE,
                               snapshot_digest="digest", sources=(), protected_paths=(),
                               current_revisions=())

    def floor(snap, pages=(), *, require_feasible=False, measure=None, violations_out=None):
        counts = [slot.identity.last_line - slot.identity.first_line + 1 for slot, _ in pages]
        size = snap.base + LINE_BYTES * sum(counts)
        violations = list(knobs["rule"](counts))
        if size > snap.byte_cap:
            violations.append("serialized_bytes")
        if violations_out is not None:
            violations_out.extend(violations)
        if require_feasible and violations:
            return None
        return SimpleNamespace(wire=b"x" * size)

    monkeypatch.setattr(live, "read_admission_snapshot", snapshot)
    monkeypatch.setattr(live, "read_admission_floor", floor)
    monkeypatch.setattr(live, "read_admission_measure", lambda runtime: None)
    monkeypatch.setattr(admission, "request_byte_cap", lambda: BASE + 10_000)
    return knobs


def test_cursor_continuation_with_one_line_room_delivers_at_least_one_line(tmp_path, envelope):
    """(a) The full continuation is infeasible; the search settles on the largest fitting prefix."""
    rt = make_runtime(tmp_path)
    write_lines(tmp_path, "b.py", 60, "b")
    cursor = read(rt, tmp_path, "b.py", {"path": "b.py", "max_lines": 10}).metadata["next_cursor"]
    rt._frame_budget_request_count += 1
    envelope["allowance"] = 250
    result = read(rt, tmp_path, "b.py", {"path": "b.py", "cursor": cursor})
    assert result.result_class == "OK"
    assert line_count(result) == 2
    assert result.metadata["next_cursor"]


def test_refused_continuation_is_carried_and_delivered_by_next_request(tmp_path, envelope):
    """(b) Siblings take the allowance first; the carried continuation still gets one line.

    Request 1 refuses the continuation (no carry yet). Request 2 keeps the carried
    1-line reserve away from the sibling, so the continuation delivers. Request 3 proves
    the carry cleared: the sibling gets the full allowance again.
    """
    rt = make_runtime(tmp_path)
    write_lines(tmp_path, "b.py", 60, "b")
    write_lines(tmp_path, "a.py", 60, "a")
    cursor = read(rt, tmp_path, "b.py", {"path": "b.py", "max_lines": 10}).metadata["next_cursor"]
    sibling_lines, continuation = [], []
    for _ in range(3):
        rt._frame_budget_request_count += 1
        sibling = read(rt, tmp_path, "a.py", {"path": "a.py"})
        sibling_lines.append(line_count(sibling))
        cont = read(rt, tmp_path, "b.py", {"path": "b.py", "cursor": cursor})
        continuation.append((cont.result_class, line_count(cont)))
    assert continuation[0][0] == admission.REFUSED_ADMISSION
    assert continuation[1] == ("OK", 1)
    assert sibling_lines == [10, 9, 10]


def test_single_line_page_is_probed_and_admitted_when_it_fits(tmp_path, monkeypatch):
    """(c) A one-line page is probed (the search bound is 1, not 0) and admitted when it fits.

    The 1-line probe is stubbed to a cheaper rendering: a real single line renders the same
    bytes as the full page, so the search bound is what this test pins.
    """
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 10_000)
    rt = make_runtime(tmp_path)
    rt.working_set_fixed_bytes = 0
    room = admission.remaining_allowance(rt, None)
    assert room >= 16
    (tmp_path / "one.py").write_text("only line\n")
    raw = (tmp_path / "one.py").read_bytes()
    request = normalize_read_request({"path": "one.py"})
    source = deliver_source(raw, "one.py", 0, args=request)
    source = SourceDelivery(result_class="OK", text='"' + "y" * (room + 500) + '"',
                            content=source.content, metadata=source.metadata)
    real = admission.deliver_source
    probes = []

    def cheaper_single_line(raw_bytes, display, start, *, args, **kwargs):
        out = real(raw_bytes, display, start, args=args, **kwargs)
        if args.max_lines == 1 and out.result_class == "OK":
            probes.append(args.max_lines)
            return SourceDelivery(result_class="OK", text='"y"', content=out.content,
                                  metadata=out.metadata)
        return out

    monkeypatch.setattr(admission, "deliver_source", cheaper_single_line)
    result = admission.admit(rt, raw=raw, display="one.py", start=0, request=request,
                             source=source, visible_messages=None)
    assert probes == [1]
    assert result.result_class == "OK" and result.text == '"y"'
    assert events(rt, decision="TRUNCATE")[0]["admitted_lines"] == 1


def test_boundary_probe_without_carrier_equality_is_admitted_when_cost_fits(tmp_path, envelope,
                                                                            monkeypatch):
    """(d) Only the 6-line prefix is feasible and no proposed carrier equals it.

    The binary search finds nothing; the carrier-boundary probe must still price it with
    cost() and admit it.
    """
    rt = make_runtime(tmp_path)
    write_lines(tmp_path, "d.py", 10, "d")
    envelope["rule"] = lambda counts: () if counts in ([], [6]) else ("prompt_tokens",)
    monkeypatch.setattr(admission, "_shared_prefix_counts", lambda state, source, path: (6,))
    result = read(rt, tmp_path, "d.py", {"path": "d.py"})
    assert result.result_class == "OK"
    assert line_count(result) == 6
    assert events(rt, decision="TRUNCATE")[0]["admitted_lines"] == 6


def test_fully_infeasible_read_refuses_with_typed_floor_diagnosis(tmp_path, envelope):
    """(e) Even one line is infeasible: refuse, journal floor_infeasible, never raise.

    The base request is infeasible too, so the base event fires exactly once per request.
    """
    rt = make_runtime(tmp_path)
    write_lines(tmp_path, "e.py", 5, "e")
    write_lines(tmp_path, "f.py", 5, "f")
    envelope["rule"] = lambda counts: ("prompt_tokens",)
    first = read(rt, tmp_path, "e.py", {"path": "e.py"})
    second = read(rt, tmp_path, "f.py", {"path": "f.py"})
    assert first.result_class == admission.REFUSED_ADMISSION
    assert second.result_class == admission.REFUSED_ADMISSION
    per_read = events(rt, refusal_cause="floor_infeasible", path="e.py")
    assert per_read
    assert per_read[0]["reason_code"] == ReasonCode.FLOOR_INFEASIBLE.value
    assert per_read[0]["violations"] == ["prompt_tokens"]
    assert len(events(rt, refusal_cause="floor_infeasible", path="")) == 1


def test_cached_sibling_cannot_spend_pending_continuation_reserve(tmp_path, monkeypatch):
    """A cached sibling obeys the same one-line reserve as a newly rendered read."""
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 350)
    rt = make_runtime(tmp_path)
    rt.working_set_fixed_bytes = 100  # 250 bytes of fresh exposure per request.
    rt._frame_budget_request_count = 1
    state = admission._state(rt, None)
    reserve = 100
    setattr(rt, admission.CARRY_ATTR, {
        "path": "b.py", "cursor": "read1:pending", "request_index": 0, "reserve_bytes": reserve,
    })
    cached_text = "x" * (state.allowance - reserve + 1)
    cached_bytes = admission._wire_bytes(cached_text)
    assert state.allowance - reserve < cached_bytes <= state.allowance
    assert not admission.fits(rt, None, cached_text)
