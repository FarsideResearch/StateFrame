"""A1b read-time admission at the read boundary (K1/K2), OFF parity and journal shape."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_read_admission as admission
import kernel_read_exec as reads
import kernel_read_receipts as receipts
import kernel_source_delivery as delivery_mod
from kernel_continuation_restart import SAFE_TAIL

GIANT = {"path": "giant.py", "max_lines": 1000, "max_chars": 64000}
CURSOR = re.compile(r"(\d+) more lines; page with cursor=(read1:[0-9a-f]{64}:\d+:\d+)")


class Journal:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []

    def emit(self, kind, **fields):
        row = dict(kind=kind, **fields)
        self.events.append(row)
        return {**row, "event_id": f"event-{len(self.events)}"}


def runtime(root, fixed=0):
    return SimpleNamespace(journal=Journal(root / "audit/events.jsonl"), retained=None,
                           step_ledger=None, _frame_budget_request_count=3,
                           working_set_fixed_bytes=fixed)


def read(rt, root, name, args=None):
    delivery = {}
    scope = receipts.begin_read_attempt(rt, args or {"path": name})
    result = reads.execute_read(root, root / name, name, name, 0, args=args or {"path": name},
                                measurement=rt, step=1, delivery=delivery)
    if scope is not None:
        scope.finish(result[0])
    return result, delivery


def big_file(root, name, lines=3000):
    text = "".join(f"line {i:05d} " + "x" * 60 + "\n" for i in range(1, lines + 1))
    (root / name).write_text(text)
    return text


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def _results(rt):
    return [e for e in rt.journal.events if e["kind"] == "read_result"]


def test_admission_refusal_settles_continuation_without_source_credit(tmp_path, on, monkeypatch):
    """A refused read has a known host outcome and must keep its cursor reply."""
    from kernel_continuation_state import ContinuationError, unresolved
    from test_no_dead_fence import _closed_case, _journal_live

    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/value.py").write_text("value = 1\n")
    (status, text, _), metadata = read(runtime(tmp_path, fixed=1000), tmp_path, "pkg/value.py")
    assert status == admission.REFUSED_ADMISSION
    assert metadata["next_cursor"] in text
    from kernel_continuation_evidence import result_class_disposition
    assert result_class_disposition("grep", status) == "unknown"
    assert result_class_disposition("text", status) == "unknown"
    from kernel_framefit_receipt_history import read_receipt
    assert read_receipt({"kind": "action_receipt", "receipt": {
        "kind": "read", "result_class": status, "target_path": "pkg/value.py",
        "receipt_id": "refused-read", "workspace_revision": "sha256:source"}}) is None
    directory = tmp_path / "continuation"
    directory.mkdir()
    journal, engine, current, closure, _ = _closed_case(
        directory, "not_applied_read", read_result=status)
    try:
        live = _journal_live(journal, engine, current, ())
        live.closed(closure)
        assert engine.state.effects["effect-1"].status == "not_applied"
        assert not unresolved(engine.state, "root")
        with pytest.raises(ContinuationError, match="does not prove a current source read"):
            engine._evidence._read(closure["event_id"], current)
    finally:
        journal.close()


def test_k1_giant_read_is_paginated_not_blocked(tmp_path, on, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 30_000)
    text = big_file(tmp_path, "giant.py")
    rt = runtime(tmp_path, fixed=10_000)  # allowance 20_000 < one 24 KB-ish page
    (result, observation, _), delivery = read(rt, tmp_path, "giant.py", GIANT)
    assert result == "OK"
    assert len(observation.encode()) <= 20_000
    match = CURSOR.search(observation)
    assert match, observation[-200:]
    delivered_end = int(delivery["shown_end_line"])
    assert int(match.group(1)) == text.count("\n") - delivered_end
    assert delivery["next_cursor"] == match.group(2)
    # The resume cursor is executable and continues exactly after the delivered range.
    (again, obs2, _), delivery2 = read(runtime(tmp_path, fixed=10_000), tmp_path, "giant.py",
                                      {"path": "giant.py", "cursor": match.group(2)})
    assert again == "OK" and int(delivery2["start_line"]) == delivered_end + 1
    # The capture covers exactly the delivered lines.
    [res] = _results(rt)
    assert res["status"] == "OK"
    assert res["identity"]["first_line"] == 1 and res["identity"]["last_line"] == delivered_end
    assert "BLOCKED" not in observation
    [event] = [e for e in rt.journal.events if e["kind"] == "read_admission"]
    assert event["decision"] == "TRUNCATE" and event["allowance"] == 20_000
    # This caller has no composed provider body: charge its JSON string bytes,
    # including escaped newlines, rather than raw UTF-8 payload bytes.
    charged = len(json.dumps(observation).encode()) - 2
    assert event["admitted_bytes"] == charged > len(observation.encode())
    assert charged <= event["allowance"]
    assert event["admitted_lines"] == delivered_end


def test_k2_batch_overflow_admits_in_order_and_refuses_the_rest_with_cursors(tmp_path, on, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 82_026)
    rt = runtime(tmp_path, fixed=4_000)  # allowance 78_026 for 16 pages of ~24 KB
    names = []
    for index in range(16):
        name = f"page{index:02d}.py"
        big_file(tmp_path, name, lines=330)  # ~ 24 KB each
        names.append(name)
    outcomes = []
    for name in names:
        (result, observation, _), delivery = read(rt, tmp_path, name, {"path": name, "max_lines": 330, "max_chars": 64000})
        outcomes.append((result, observation, delivery))
    kinds = [r for r, _, _ in outcomes]
    assert kinds[:3] == ["OK", "OK", "OK"], kinds
    assert "BLOCKED" not in "".join(o for _, o, _ in outcomes)
    admitted = sum(len(o.encode()) for r, o, _ in outcomes if r == "OK")
    assert admitted <= 78_026
    refused = [(o, d) for r, o, d in outcomes if r == admission.REFUSED_ADMISSION]
    assert refused and kinds[-1] == admission.REFUSED_ADMISSION
    # Admission is in batch order: once refused, every later call is refused.
    first_refused = kinds.index(admission.REFUSED_ADMISSION)
    assert all(k == admission.REFUSED_ADMISSION for k in kinds[first_refused:])
    for observation, delivery in refused:
        match = CURSOR.fullmatch(observation)
        assert match and int(match.group(1)) == 330 and delivery["next_cursor"] == match.group(2)
        assert delivery["result_class"] == admission.REFUSED_ADMISSION
    # Nothing is dropped: every read has exactly one typed read_result.
    results = _results(rt)
    assert len(results) == 16
    assert [r["status"] for r in results] == kinds
    events = [e for e in rt.journal.events if e["kind"] == "read_admission"]
    assert len(events) == 16
    assert events[-1]["refused_count"] == len(refused)
    assert events[-1]["admitted_count"] + events[-1]["truncated_count"] + events[-1]["refused_count"] == 16
    # A truncated page in the middle still ends on a cursor with a truthful count.
    for result, observation, delivery in outcomes:
        if result == "OK" and delivery.get("truncated"):
            match = CURSOR.search(observation)
            assert match and int(match.group(1)) == 330 - int(delivery["shown_end_line"])


def test_refused_resume_cursor_is_executable(tmp_path, on, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1_000)
    big_file(tmp_path, "src.py", lines=50)
    rt = runtime(tmp_path, fixed=1_000)  # allowance 0: nothing fits
    (result, observation, _), delivery = read(rt, tmp_path, "src.py", {"path": "src.py", "line": 21})
    assert result == admission.REFUSED_ADMISSION
    match = CURSOR.fullmatch(observation)
    assert match and int(match.group(1)) == 30
    [res] = _results(rt)
    assert res["status"] == admission.REFUSED_ADMISSION and "payload_blob_ref" not in res
    (again, _, _), delivery2 = read(runtime(tmp_path, fixed=0), tmp_path, "src.py",
                                   {"path": "src.py", "cursor": match.group(2)})
    assert again == "OK" and int(delivery2["start_line"]) == 21


def test_pending_allowance_resets_per_request(tmp_path, on, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1_000)
    (tmp_path / "a.txt").write_text("alpha\n")
    rt = runtime(tmp_path, fixed=0)
    read(rt, tmp_path, "a.txt")
    assert admission.remaining_allowance(rt, None) < 1_000
    rt._frame_budget_request_count = 4
    assert admission.remaining_allowance(rt, None) == 1_000


def test_fixed_estimate_is_messages_plus_reserve_and_pin_wins():
    messages = [{"role": "user", "content": "x" * 100}]
    rt = SimpleNamespace()
    estimate = admission.fixed_envelope_estimate(rt, messages)
    assert estimate > 100 + admission.FIXED_RESERVE_BYTES
    # No messages and no sealed measurement: the allowance is the unmeasured default, not cap-8K.
    assert admission.fixed_envelope_estimate(SimpleNamespace(), None) == (
        admission.request_byte_cap() - admission.UNMEASURED_FRESH_ALLOWANCE_BYTES)
    assert admission.fixed_envelope_estimate(SimpleNamespace(working_set_fixed_bytes=7), messages) == 7


def test_off_is_byte_identical_and_emits_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1_000)
    big_file(tmp_path, "giant.py")
    rt = runtime(tmp_path, fixed=1_000)
    (result, observation, _), delivery = read(rt, tmp_path, "giant.py", GIANT)
    assert result == "OK"
    expected = delivery_mod.deliver_source((tmp_path / "giant.py").read_bytes(), "giant.py", 0,
                                           args=GIANT)
    assert observation == expected.text
    assert not [e for e in rt.journal.events if e["kind"] == "read_admission"]
    assert not hasattr(rt, "_read_admission_state")


def test_admission_event_is_safe_tail_and_status_is_typed():
    assert admission.EVENT in SAFE_TAIL
    assert receipts.typed_status(admission.REFUSED_ADMISSION) == admission.REFUSED_ADMISSION


def test_cached_page_that_overflows_is_rerendered_shorter(tmp_path, on, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 30_000)
    big_file(tmp_path, "giant.py")
    rt = runtime(tmp_path, fixed=0)  # first read: everything fits and the page is cached
    (first, full, _), _ = read(rt, tmp_path, "giant.py", GIANT)
    assert first == "OK" and len(full.encode()) > 20_000
    rt._frame_budget_request_count = 4
    rt.working_set_fixed_bytes = 10_000
    (second, shorter, _), delivery = read(rt, tmp_path, "giant.py", GIANT)
    assert second == "OK" and len(shorter.encode()) <= 20_000 and CURSOR.search(shorter)
    [_, res] = _results(rt)
    assert res["identity"]["last_line"] == int(delivery["shown_end_line"])


# --- r2 review fixes: measured envelope, unmeasured callers, cached replay charge ---

def _bare_runtime(root):
    rt = runtime(root)
    del rt.working_set_fixed_bytes
    return rt


def test_sealed_request_measurement_wins_over_messages_estimate(tmp_path):
    rt = _bare_runtime(tmp_path)
    owner = [{"role": "system", "content": "s" * 1000}, {"role": "user", "content": "u" * 2000}]
    # Request composition adds what the read boundary cannot see: schemas, frame, windows.
    wire = owner + [{"role": "user", "content": "[StateFrame]" + "f" * 18_000},
                    {"role": "tool", "tool_call_id": "c1", "content": "[read page] " + "p" * 3_000}]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"x": "t" * 17_000}}}]
    sealed = admission.note_sealed_request(rt, wire, tools)
    import json
    expected = (len(json.dumps({"messages": wire, "tools": tools}).encode())
                - len(json.dumps(wire[-1]).encode()))
    assert sealed == expected and admission.sealed_fixed_bytes(rt) == expected
    assert sealed > 38_000  # the fresh reply is the only thing not counted
    assert admission.fixed_envelope_estimate(rt, owner) == expected
    # The messages estimate still wins when it is the larger one.
    huge = owner + [{"role": "user", "content": "h" * 60_000}]
    assert admission.fixed_envelope_estimate(rt, huge) > expected
    # A pinned value overrides both.
    rt.working_set_fixed_bytes = 7
    assert admission.fixed_envelope_estimate(rt, huge) == 7


def test_recorded_wire_envelope_is_measured_not_guessed():
    wire_path = Path(os.environ.get("STATEFRAME_RECORDED_WIRE") or "/nonexistent/recorded-wire.json")
    if not wire_path.exists():
        pytest.skip("recorded wire not present")
    import json
    body = json.loads(wire_path.read_text())
    rt = SimpleNamespace()
    sealed = admission.note_sealed_request(rt, body["messages"], body["tools"])
    owner = [m for m in body["messages"] if m["role"] in {"system", "assistant"}]
    guess = len(json.dumps(owner, ensure_ascii=False, separators=(",", ":")).encode()) + admission.FIXED_RESERVE_BYTES
    assert sealed > 60_000 > guess
    assert admission.fixed_envelope_estimate(rt, owner) == sealed


def test_unmeasured_caller_gets_two_frame_pages_not_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 82_026)
    rt = _bare_runtime(tmp_path)
    assert admission.fixed_envelope_estimate(rt, None) == 82_026 - admission.UNMEASURED_FRESH_ALLOWANCE_BYTES
    assert admission.fixed_envelope_estimate(rt, []) == 82_026 - admission.UNMEASURED_FRESH_ALLOWANCE_BYTES
    assert admission.remaining_allowance(rt, None) == admission.UNMEASURED_FRESH_ALLOWANCE_BYTES


def test_note_provider_request_records_sealed_bytes_only_when_on(tmp_path, monkeypatch):
    from kernel_stateframe_runtime import note_provider_request
    msgs = [{"role": "user", "content": "hello"}, {"role": "tool", "tool_call_id": "c", "content": "x" * 50}]
    tools = [{"type": "function", "function": {"name": "read"}}]
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    rt = SimpleNamespace(_frame_budget_request_count=0)
    note_provider_request(rt, msgs, tools)
    assert rt._frame_budget_request_count == 1
    assert admission.sealed_fixed_bytes(rt) == admission.note_sealed_request(SimpleNamespace(), msgs, tools)
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    off = SimpleNamespace(_frame_budget_request_count=0)
    note_provider_request(off, msgs, tools)
    assert off._frame_budget_request_count == 1 and admission.sealed_fixed_bytes(off) is None


def test_cached_replay_is_charged_against_the_batch(tmp_path, on, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 60_000)
    big_file(tmp_path, "a.py", lines=330)
    big_file(tmp_path, "b.py", lines=330)
    rt = runtime(tmp_path, fixed=10_000)  # allowance 50_000: two ~24 KB pages fit, three do not
    args_a = {"path": "a.py", "max_lines": 330, "max_chars": 64000}
    args_b = {"path": "b.py", "max_lines": 330, "max_chars": 64000}
    (r1, o1, _), _ = read(rt, tmp_path, "a.py", args_a)
    assert r1 == "OK"
    # Next request: the source bytes are dropped, the rendered page stays replayable.
    for entry in rt._read_contract_source_cache.values():
        entry.content = None
    rt._frame_budget_request_count = 4
    (r2, o2, _), d2 = read(rt, tmp_path, "a.py", args_a)
    (r3, o3, _), _ = read(rt, tmp_path, "a.py", args_a)
    assert r2 == r3 == "OK" and d2.get("read_dedup", {}).get("status") == "CACHED_BYTES"
    events = [e for e in rt.journal.events if e["kind"] == "read_admission" and e["request_index"] == 4]
    assert [e["decision"] for e in events] == ["CACHED", "CACHED"], events
    assert o1 == o2 == o3
    charged = len(json.dumps(o1).encode()) - 2
    assert charged > len(o1.encode())
    assert [e["admitted_bytes"] for e in events] == [charged, charged]
    assert events[-1]["pending_before"] == charged
    assert admission.remaining_allowance(rt, None) == 50_000 - 2 * charged
    # The fresh third page now overflows and is shortened; without the charge it would be whole.
    (r4, o4, _), d4 = read(rt, tmp_path, "b.py", args_b)
    assert r4 == "OK" and d4.get("truncated") and CURSOR.search(o4)
    assert 2 * charged + len(json.dumps(o4).encode()) - 2 <= 50_000
