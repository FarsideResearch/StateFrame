"""Range-scoped read-fact truth on the final wire (S6 / A3).

A per-file fact is scoped to the carrier actually on the wire: it says
"lines a-b shown; the rest not shown" and never "not shown" for a range that
is attached. Plan-only proposals confer nothing and take nothing away, and
history compaction (keep-1 receipts, frame-side historical_receipts) never
demotes a present fact.
"""
import hashlib
import json
import re
from dataclasses import replace

import pytest
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_prepare import _compact_history
from kernel_working_set_render import domain, render_assignment
from test_working_set_render import snapshot, source

HOST_FACT = re.compile(r"(\S+) lines (\d+)-(\d+) revision ([^;\s]+)")
SHOWN = re.compile(r"lines (\d+)-(\d+) shown")


def _wire_truth(wire):
    """The reviewer's wire-truth predicate: one range is never both shown and not shown.

    Returns the read facts found on the wire after asserting each one against
    the FULL carriers the same wire attaches.
    """
    messages = json.loads(wire)["messages"]
    attached = {}
    for message in messages:
        content = message.get("content", "")
        if content.startswith("[Host fact]\nCurrent source bytes attached"):
            for path, first, last, revision in HOST_FACT.findall(content.split("\n", 1)[1]):
                attached.setdefault((path, revision), []).append((int(first), int(last)))
    facts = []
    for message in messages:
        content = message.get("content", "")
        if not content.startswith("[StateFrame]\n"):
            continue
        state = json.loads(content.split("\n", 1)[1])["state"]
        for fact in state.get("framefit5", {}).get("read_facts", ()):
            on_wire = attached.get((fact["path"], fact["read_revision"]), [])
            attached_lines = {line for a, b in on_wire for line in range(a, b + 1)}
            notice = fact.get("delivery_notice", "")
            if fact["body_delivery"] == "not_shown":
                assert not on_wire, f"{fact['path']} is on the wire but claimed not shown: {notice}"
                assert notice.startswith("[not shown] ")
            elif fact["body_delivery"] == "present":
                assert on_wire and "delivery_notice" not in fact
                lines = fact["lines"]
                assert set(range(lines["start"], lines["end"] + 1)) <= attached_lines
            else:
                assert fact["body_delivery"] == "partial"
                claimed = [(int(a), int(b)) for a, b in SHOWN.findall(notice)]
                assert claimed and "; the rest not shown" in notice, notice
                for first, last in claimed:
                    assert set(range(first, last + 1)) <= attached_lines, (notice, on_wire)
            facts.append(fact)
    return facts


def _frame_body(slot, frame):
    body = json.loads(snapshot(slot).fixed_body)
    body["messages"].insert(2, {"role": "user", "content": "[StateFrame]\n" + json.dumps(
        {"state": {"framefit5": frame}})})
    return body


def _fact(slot, receipts, **extra):
    return {"path": "a.py", "read_revision": slot.identity.revision,
            "current_revision": slot.identity.revision,
            "lines": {"start": slot.identity.first_line, "end": slot.identity.last_line, "total": 99},
            "receipt_count": receipts, "body_delivery": "present",
            "receipts": [{"receipt_id": f"c{i}", "lines": {"start": 1, "end": 40}}
                         for i in range(receipts)], **extra}


def _captured(text="x = 1\n" * 40):
    return replace(source("a", text, state="RESIDENT"), capture_ids=("c0", "c1"))


@pytest.mark.parametrize("minimum", [False, True])
def test_keep_one_receipt_compaction_never_demotes_present_fact(minimum):
    slot = _captured()
    frame = {"read_facts": [_fact(slot, 2)]}
    _compact_history(frame, minimum=minimum)
    fact = frame["read_facts"][0]
    assert fact["historical_receipts"]["count"] == 1 and fact["receipt_count"] == 2
    snap = snapshot(slot, body=_frame_body(slot, frame),
                    current_revisions=(("a.py", slot.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {slot.key: FULL}).wire)
    assert fact["body_delivery"] == "present"


def test_frame_side_historical_receipts_do_not_demote_present_fact():
    slot = _captured()
    fact = _fact(slot, 1)
    fact["receipt_count"] = 2
    fact["historical_receipts"] = {"count": 1, "sha256": "0" * 64}
    snap = snapshot(slot, body=_frame_body(slot, {"read_facts": [fact]}),
                    current_revisions=(("a.py", slot.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {slot.key: FULL}).wire)
    assert fact["body_delivery"] == "present"


def test_deferred_plan_only_proposal_does_not_demote_captured_full_fact():
    slot = _captured()
    plan = replace(source("closure:a", "x = 1\n" * 99, state="CLOSURE", ordinal=2),
                   capture_ids=(), plan_only=True)
    snap = snapshot(slot, plan, body=_frame_body(slot, {"read_facts": [_fact(slot, 2)]}),
                    current_revisions=(("a.py", slot.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {slot.key: FULL, plan.key: DEFERRED}).wire)
    assert fact["body_delivery"] == "present"


def test_disjoint_deferred_capture_scopes_fact_to_the_attached_range():
    first = replace(source("first", "one\n", first=1, state="RESIDENT"), capture_ids=("c0",))
    second = replace(source("second", "two\n", first=20, ordinal=2, state="RESIDENT"),
                     capture_ids=("c1",))
    fact = _fact(first, 2)
    snap = snapshot(first, second, body=_frame_body(first, {"read_facts": [fact]}),
                    current_revisions=(("a.py", first.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {"first": FULL, "second": DEFERRED}).wire)
    assert fact["body_delivery"] == "partial"
    assert fact["delivery_notice"] == (
        f"[partly shown] a.py @rev {first.identity.revision}: lines 1-1 shown; "
        "the rest not shown: re-read to see it")
    [fact] = _wire_truth(render_assignment(snap, {"first": FULL, "second": FULL}).wire)
    assert fact["body_delivery"] == "present"


def test_retired_receipt_scopes_fact_instead_of_denying_the_attached_range():
    slot = replace(source("new", "one\n", state="RESIDENT"), capture_ids=("new-capture",))
    fact = _fact(slot, 2)
    fact["receipts"] = [{"receipt_id": "old-retired"}, {"receipt_id": "new-capture"}]
    snap = snapshot(slot, body=_frame_body(slot, {"read_facts": [fact]}),
                    current_revisions=(("a.py", slot.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {slot.key: FULL}).wire)
    assert fact["body_delivery"] == "partial"
    assert "lines 1-1 shown; the rest not shown" in fact["delivery_notice"]


def test_deferred_only_fact_stays_not_shown_and_stale_revision_is_not_scoped():
    slot = _captured()
    snap = snapshot(slot, body=_frame_body(slot, {"read_facts": [_fact(slot, 2)]}),
                    current_revisions=(("a.py", slot.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {slot.key: DEFERRED}).wire)
    assert fact["body_delivery"] == "not_shown"
    stale = snapshot(slot, body=_frame_body(slot, {"read_facts": [_fact(slot, 2)]}),
                     current_revisions=(("a.py", "sha256:" + "2" * 64),))
    # G1 retires obsolete residency; G14 must not demand an illegal FULL mode.
    assert domain(stale, slot) == (DEFERRED,)
    with pytest.raises(ValueError, match="frozen slot domain"):
        render_assignment(stale, {slot.key: FULL})
    [fact] = _wire_truth(render_assignment(stale, {slot.key: DEFERRED}).wire)
    assert fact["body_delivery"] == "not_shown"
    # Unexposed DUE debt is legitimately FULL even after an edit. Preserve G14's
    # stale-FULL control through that carrier, without claiming current source.
    due = replace(slot, state="DUE")
    stale_due = replace(stale, sources=(due,))
    assert domain(stale_due, due) == (FULL,)
    [fact] = _wire_truth(render_assignment(stale_due, {due.key: FULL}).wire)
    assert fact["body_delivery"] == "not_shown"


def test_partly_shown_notice_is_carried_into_the_capsule_and_rewritten_again():
    first = replace(source("first", "one\n", first=1, state="RESIDENT"), capture_ids=("c0",))
    second = replace(source("second", "two\n", first=20, ordinal=2, state="RESIDENT"),
                     capture_ids=("c1",))
    capsule = ("earlier exchanges trimmed to fit context. CAPSULE\nWORLD KNOWLEDGE now\n"
               "- a.py lines 1-20 | fresh | read@3\n+ next\n")
    body = _frame_body(first, {"read_facts": [_fact(first, 2)], "prior_user_context": [
        {"content": capsule, "content_sha256": "x"}]})
    snap = snapshot(first, second, body=body,
                    current_revisions=(("a.py", first.identity.revision),))
    wire = render_assignment(snap, {"first": FULL, "second": DEFERRED}).wire
    [fact] = _wire_truth(wire)
    frame = next(json.loads(m["content"].split("\n", 1)[1])["state"]["framefit5"]
                 for m in json.loads(wire)["messages"]
                 if m.get("content", "").startswith("[StateFrame]\n"))
    line = frame["prior_user_context"][0]["content"]
    assert f"| {fact['delivery_notice']} | read@3" in line
    assert frame["prior_user_context"][0]["content_sha256"] == hashlib.sha256(line.encode()).hexdigest()
    # A later rebuild must recognise and replace the partly-shown label too.
    from kernel_working_set_facts import _update_frame
    frame["read_facts"][0]["body_delivery"] = "present"
    _update_frame(frame, snap, [first, second])
    assert "| fresh | read@3" in frame["prior_user_context"][0]["content"]
    row = frame["prior_user_context"][0]
    assert row["content_sha256"] == hashlib.sha256(row["content"].encode()).hexdigest()


@pytest.mark.parametrize("second_start,expected", [
    (2, "lines 1-3 shown"),
    (3, "lines 1-4 shown"),
    (5, "lines 1-2 shown, lines 5-6 shown"),
])
def test_partial_notice_merges_only_overlapping_or_adjacent_ranges(second_start, expected):
    first = source("first", "one\ntwo\n", state="RESIDENT")
    second = source("second", "two\nthree\n", first=second_start, state="RESIDENT", ordinal=2)
    omitted = source("omitted", "missing\n", first=20, state="RESIDENT", ordinal=3)
    snap = snapshot(first, second, omitted,
                    body=_frame_body(first, {"read_facts": [_fact(first, 3)]}),
                    current_revisions=(("a.py", first.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {
        first.key: FULL, second.key: FULL, omitted.key: DEFERRED}).wire)
    assert fact["body_delivery"] == "partial"
    assert fact["delivery_notice"].endswith(f": {expected}; the rest not shown: re-read to see it")


def test_earlier_disjoint_interval_does_not_hide_a_fully_shown_later_fact():
    first = source("first", "earlier\n", state="RESIDENT")
    later = source("later", "later\n", first=20, state="RESIDENT", ordinal=2)
    snap = snapshot(first, later, body=_frame_body(first, {"read_facts": [_fact(later, 2)]}),
                    current_revisions=(("a.py", first.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {first.key: FULL, later.key: FULL}).wire)
    assert fact["body_delivery"] == "present"


def test_clipped_carrier_is_not_promoted_to_whole_line_coverage():
    clipped = source("clipped", "cut", state="RESIDENT")
    clipped = replace(clipped, page=clipped.page.replace("lines=1-1/99", "lines=1-1/99 clip=max_chars"))
    whole = source("whole", "whole\n", first=5, state="RESIDENT", ordinal=2)
    snap = snapshot(clipped, whole, body=_frame_body(clipped, {"read_facts": [_fact(clipped, 2)]}),
                    current_revisions=(("a.py", clipped.identity.revision),))
    [fact] = _wire_truth(render_assignment(snap, {clipped.key: FULL, whole.key: FULL}).wire)
    assert fact["body_delivery"] == "partial"
    assert fact["delivery_notice"].endswith(": lines 5-5 shown; the rest not shown: re-read to see it")


def test_off_provider_bytes_match_group_base(tmp_path, tmp_path_factory):
    from test_working_set_off_parity_base import (
        test_off_provider_bytes_match_cutover_base,
    )

    test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, "1f6b7e9bc1")
