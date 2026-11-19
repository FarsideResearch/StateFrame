"""Complete shadow candidates: ownership, coupled costs and hostile certificates."""
from dataclasses import replace
import hashlib
import json

import pytest

from kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
from kernel_working_set_render import render_assignment, propose_slots
from kernel_working_set_proof import CertificateError, verify_certificate
from kernel_working_set_prepare import prepare_candidate, PreparedCandidate, Blocked
from kernel_working_set_allocator import FULL, DEFERRED
from kernel_working_set_stop import ReasonCode


def source(key="a", text='hello \\" λ\n', *, first=1, revision=None, state="DUE", ordinal=1):
    digest = hashlib.sha256(text.encode()).hexdigest()
    identity = ReceiptIdentity("a.py", revision or "sha256:" + "1" * 64,
                               first, first + len(text.splitlines()) - 1,
                               0, len(text.encode()), digest, None)
    digests, numbers = line_digests_with_numbers(text, first)
    page = f"[read page] path=a.py revision={identity.revision} lines={first}-{identity.last_line}/99\n0 more; complete=true\n{text}"
    return SourceSlot(key, (key,), identity, text, page, digests, numbers,
                      b'{"path":"a.py","start_line":1,"end_line":99}',
                      ordinal, ordinal, state)


def snapshot(*slots, body=None, cap=100_000, **kwargs):
    body = body or {"model": "local", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "opaque newest suffix"}], "max_tokens": 100}
    return PreparationSnapshot(json.dumps(body).encode(), tuple(slots),
                               "request", 1, cap, **kwargs)


def test_escaping_order_and_exact_recompute():
    a, b = source("a", ordinal=2), source("b", "other\n", first=10)
    snap = snapshot(a, b)
    candidate = render_assignment(snap, {"a": FULL, "b": FULL}, "plain")
    body = json.loads(candidate.wire)
    assert candidate.wire == json.dumps(body).encode()
    assert candidate.wire == render_assignment(snap, {"b": FULL, "a": FULL}, "plain").wire
    contents = [m.get("content", "") for m in body["messages"]]
    assert contents.index(b.page) < contents.index(a.page) < contents.index("opaque newest suffix")
    assert sum(dict(candidate.section_breakdown).values()) == len(candidate.wire)
    assert verify_certificate(candidate.wire, candidate.certificate)


def test_disjoint_ranges_and_historical_revision_never_share_authority():
    a = source("a", "one\n", state="RESIDENT")
    b = source("b", "two\n", first=20, revision="sha256:" + "2" * 64)
    candidate = render_assignment(snapshot(a, b), {"a": DEFERRED, "b": FULL}, "plain")
    assert [w.mode for w in candidate.certificate.witnesses] == ["DEFERRED", "FULL"]
    assert verify_certificate(candidate.wire, candidate.certificate)
    forged = replace(candidate.certificate, witnesses=(candidate.certificate.witnesses[1],) * 2)
    with pytest.raises(CertificateError):
        verify_certificate(candidate.wire, forged)


def test_read_admission_protects_overlaps_and_keeps_pending_edit_paths():
    from kernel_source_delivery import deliver_source
    from kernel_working_set_cache import transaction
    from kernel_working_set_live import read_admission_floor, read_admission_slot
    from kernel_working_set_render import domain, snapshot_digest

    raw = b"line\n" * 30
    args = {"path": "a.py", "start_line": 10, "end_line": 11}
    proposal = read_admission_slot(deliver_source(raw, "a.py", 0, args=args),
                                   path="a.py", arguments=args, ordinal=3)
    revision = proposal.identity.revision
    overlapping = source("overlap", "line\n" * 3, first=9, state="RESIDENT", revision=revision)
    disjoint = source("disjoint", "line\n" * 3, first=12, state="RESIDENT", revision=revision, ordinal=2)
    base = snapshot(overlapping, disjoint)

    @transaction
    def check():
        assert DEFERRED in domain(base, overlapping)
        floor = read_admission_floor(base, [(proposal, None)])
        admitted = floor.certificate.snapshot
        assert admitted.protected_paths == ()
        assert admitted.protected_ranges == (("a.py", 10, 11),)
        assert domain(admitted, overlapping) == (FULL,)
        assert DEFERRED in domain(admitted, disjoint)
        assert snapshot_digest(admitted) != snapshot_digest(replace(admitted, protected_ranges=()))
        assert admitted.snapshot_digest != replace(admitted, protected_ranges=()).snapshot_digest
        edit = replace(base, protected_paths=("a.py",))
        edit_floor = read_admission_floor(edit, [(proposal, None)])
        assert edit_floor.certificate.snapshot.protected_paths == ("a.py",)
        assert domain(edit_floor.certificate.snapshot, disjoint) == (FULL,)

    check()


def test_shared_capture_carrier_and_newest_exchange():
    a = source()
    b = replace(a, key="b", obligation_ids=("b",), admission_ordinal=2)
    snap = snapshot(a, b)
    result = render_assignment(snap, {"a": FULL, "b": FULL}, "plain")
    assert [m["content"] for m in json.loads(result.wire)["messages"]].count(a.page) == 1
    assert result.certificate.witnesses[0].message_index == result.certificate.witnesses[1].message_index
    assert verify_certificate(result.wire, result.certificate)
    body = json.loads(snap.fixed_body)
    body["messages"].append({"role": "tool", "content": a.page})
    result = render_assignment(snapshot(a, body=body), {"a": FULL}, "plain")
    assert [m["content"] for m in json.loads(result.wire)["messages"]].count(a.page) == 1
    assert verify_certificate(result.wire, result.certificate)


def test_full_capture_shares_tool_page_with_a_host_annotation():
    a = source(text="a long captured line " * 500 + "\n")
    annotated = a.page + "\n[read] current requested content is in this result."
    body = json.loads(snapshot(a).fixed_body)
    body["messages"].append({"role": "tool", "content": annotated})
    result = render_assignment(snapshot(a, body=body), {a.key: FULL})
    messages = json.loads(result.wire)["messages"]
    assert sum(a.payload in str(row.get("content", "")) for row in messages) == 1
    witness, = result.certificate.witnesses
    assert messages[witness.message_index]["role"] == "tool"
    assert verify_certificate(result.wire, result.certificate)
    prepared = prepare_candidate(snapshot(a, body=body, cap=len(result.wire)))
    assert isinstance(prepared, PreparedCandidate)
    assert prepared.wire == result.wire


def test_forged_notice_or_reformatted_wire_rejected():
    a = source(state="RESIDENT")
    result = render_assignment(snapshot(a), {"a": DEFERRED}, "plain")
    assert verify_certificate(result.wire, result.certificate)
    for wire in (result.wire.replace(b"Source page deferred", b"Source page forged!!"),
                 json.dumps(json.loads(result.wire), separators=(",", ":")).encode()):
        with pytest.raises(CertificateError):
            verify_certificate(wire, result.certificate)


def test_count_digit_growth_is_in_whole_candidate_cost():
    slots = tuple(source(str(i), f"value{i}\n", first=i + 1, ordinal=i + 1, state="RESIDENT") for i in range(10))
    snap = snapshot(*slots)
    result = prepare_candidate(snap)
    assert isinstance(result, PreparedCandidate)
    assert result.total_bytes == len(result.rendered.wire)
    assert len(propose_slots(snap)) == 10
    assert verify_certificate(result.rendered.wire, result.rendered.certificate)


def test_floor_first_blocks_without_token_measurements():
    snap = snapshot(source(), cap=1)
    calls = []
    result = prepare_candidate(snap, measure=lambda wire: calls.append(wire))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    assert result.floor_bytes > snap.byte_cap
    result.proof.validate()
    assert not calls


def test_token_loop_bound_and_reserved_floor():
    slots = tuple(source(str(i), "x" * 500 + "\n", first=i + 1, ordinal=i + 1, state="RESIDENT") for i in range(12))
    snap = snapshot(*slots, prompt_limit=100)
    calls = []
    def measure(wire):
        calls.append(wire)
        return 101, 0, "test"
    result = prepare_candidate(snap, measure=measure)
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED
    assert 1 <= len(calls) <= 8
    assert len(calls[-1]) == result.floor_bytes


def test_fresh_and_protected_slots_are_full_only():
    a, b = source(), source("b", state="RESIDENT")
    slots = propose_slots(snapshot(a, b, protected_paths=("a.py",)))
    assert all(tuple(mode for mode, cost in s.domain) == (FULL,) for s in slots)
    with pytest.raises(ValueError):
        render_assignment(snapshot(a), {"a": DEFERRED}, "plain")
