"""Host-owned spans, rather than paths or notice prose, authorize delivery."""
import hashlib
import json
from dataclasses import replace

import pytest
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_proof import CertificateError, verify_certificate
from kernel_working_set_render import render_assignment
from test_working_set_render import snapshot, source


def test_same_payload_at_disjoint_ranges_requires_distinct_carriers():
    first = source("first", "same\n", first=1)
    second = source("second", "same\n", first=19, ordinal=2)
    result = render_assignment(snapshot(first, second), {"first": FULL, "second": FULL})
    a, b = result.certificate.witnesses
    assert a.message_index != b.message_index
    # The bytes agree, but the second receipt's carrier is still owned by the
    # first interval. Rebinding its coordinates must not prove the second one.
    forged = replace(result.certificate, witnesses=(a, replace(b,
        message_index=a.message_index, content_end=a.content_end,
        carrier_sha256=a.carrier_sha256, payload_start=a.payload_start,
        payload_end=a.payload_end)))
    with pytest.raises(CertificateError):
        verify_certificate(result.wire, forged)


def test_historical_full_does_not_claim_current_source():
    old = source("old", "old\n", revision="sha256:" + "1" * 64)
    current = source("new", "new\n", first=10, revision="sha256:" + "2" * 64,
                     state="RESIDENT", ordinal=2)
    snap = snapshot(old, current, current_revisions=(("a.py", current.identity.revision),))
    result = render_assignment(snap, {"old": FULL, "new": DEFERRED})
    body = json.loads(result.wire)
    host = "\n".join(m["content"] for m in body["messages"]
                     if m.get("content", "").startswith("[Host fact]"))
    assert "current source" not in host
    assert any(old.payload in m.get("content", "") for m in body["messages"])
    assert verify_certificate(result.wire, result.certificate)


def test_notice_is_not_transferable_to_another_obligation():
    first = source("first", "one\n", state="RESIDENT")
    second = source("second", "two\n", state="RESIDENT", first=20, ordinal=2)
    result = render_assignment(snapshot(first, second), {"first": DEFERRED, "second": DEFERRED})
    a, b = result.certificate.witnesses
    forged = replace(result.certificate, witnesses=(a, replace(b,
        message_index=a.message_index, content_end=a.content_end,
        carrier_sha256=a.carrier_sha256)))
    with pytest.raises(CertificateError):
        verify_certificate(result.wire, forged)


def test_modified_payload_with_rebound_outer_hash_still_fails():
    slot = source(text="good\n")
    result = render_assignment(snapshot(slot), {slot.key: FULL})
    wire = result.wire.replace(b"good", b"evil")
    certificate = replace(result.certificate, body_sha256=hashlib.sha256(wire).hexdigest())
    with pytest.raises(CertificateError):
        verify_certificate(wire, certificate)


def test_frozen_snapshot_is_independent_of_environment(monkeypatch):
    slot = source()
    snap = snapshot(slot)
    before = render_assignment(snap, {slot.key: FULL})
    monkeypatch.setenv("STATEFRAME_REASONING_MODE", "off")
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    after = render_assignment(snap, {slot.key: FULL})
    assert before == after
    assert verify_certificate(after.wire, after.certificate)


def test_normalization_precedes_hash_and_byte_cost():
    slot = source()
    body = json.loads(snapshot(slot).fixed_body)
    body["max_tokens"] = 10_000
    body["messages"][0]["__private_test"] = "hidden"
    snap = snapshot(slot, body=body, context_limit=40_000, prompt_limit=33_344)
    result = render_assignment(snap, {slot.key: FULL})
    normalized = json.loads(result.wire)
    assert normalized["max_tokens"] <= 40_000 - 33_344
    assert all(not key.startswith("__") for row in normalized["messages"] for key in row)
    assert verify_certificate(result.wire, result.certificate)


def test_deferred_assignment_rebuilds_frame_read_truth_and_source_access():
    slot = source(state="RESIDENT")
    frame = {"state": {"framefit5": {
        "read_facts": [{"path": "a.py", "read_revision": slot.identity.revision,
                        "current_revision": slot.identity.revision,
                        "lines": {"start": 1, "end": 1}, "body_delivery": "present"}],
        "next_site": {"path": "a.py", "revision": slot.identity.revision,
                      "lines": {"start": 1, "end": 1},
                      "current_bytes": slot.payload, "read_this_request": True},
    }}}
    snap = snapshot(slot)
    body = json.loads(snap.fixed_body)
    body["messages"].insert(2, {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame)})
    result = render_assignment(snapshot(slot, body=body), {slot.key: DEFERRED})
    messages = json.loads(result.wire)["messages"]
    state = json.loads(next(m["content"] for m in messages
                           if m.get("content", "").startswith("[StateFrame]\n")).split("\n", 1)[1])
    rendered = state["state"]["framefit5"]
    assert rendered["read_facts"][0]["body_delivery"] == "not_shown"
    assert rendered["next_site"]["current_bytes"] is None
    assert rendered["next_site"]["read_this_request"] is False


def test_cache_overlay_reconstructs_and_proves_physical_carriers():
    from kernel_cache_layout_view import semantic_messages
    from test_final_cache_layout import _body
    slot = source()
    snap = snapshot(slot, body=_body(exchange=True), layouts=("plain", "final_cache"), insertion_index=2)
    plain = render_assignment(snap, {slot.key: FULL}, "plain")
    cache = render_assignment(snap, {slot.key: FULL}, "final_cache")
    assert semantic_messages(json.loads(cache.wire)["messages"]) == json.loads(plain.wire)["messages"]
    assert verify_certificate(cache.wire, cache.certificate)


def test_full_range_does_not_prove_disjoint_per_path_read_fact():
    first = source("first", "one\n", first=1, state="RESIDENT")
    second = source("second", "two\n", first=20, ordinal=2, state="RESIDENT")
    frame = {"state": {"framefit5": {"read_facts": [{
        "path": "a.py", "read_revision": first.identity.revision,
        "current_revision": first.identity.revision,
        "lines": {"start": 1, "end": 1}, "body_delivery": "present"}]}}}
    body = json.loads(snapshot(first).fixed_body)
    body["messages"].insert(2, {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame)})
    snap = snapshot(first, second, body=body, current_revisions=(("a.py", first.identity.revision),))
    # Range-scoped: the attached 1-1 range is shown, the deferred 20-20 is not.
    for modes, expected in [({"first": FULL, "second": DEFERRED}, "partial"),
                            ({"first": FULL, "second": FULL}, "present")]:
        result = render_assignment(snap, modes)
        rendered = next(m["content"] for m in json.loads(result.wire)["messages"]
                        if m.get("content", "").startswith("[StateFrame]\n"))
        assert json.loads(rendered.split("\n", 1)[1])["state"]["framefit5"]["read_facts"][0]["body_delivery"] == expected


def test_unknown_current_revision_never_mints_current_source_claim():
    slot = source()
    result = render_assignment(snapshot(slot), {slot.key: FULL})
    assert not any("Current source bytes attached" in m.get("content", "")
                   for m in json.loads(result.wire)["messages"])


def test_source_access_requires_every_same_path_interval():
    first = source("first", "one\n", first=1, state="RESIDENT")
    second = source("second", "two\n", first=20, ordinal=2, state="RESIDENT")
    body = json.loads(snapshot(first).fixed_body)
    frame = {"state": {"source_access": {"path": "a.py", "included": True,
        "no_read_call_required_for_this_included_source": True}}}
    body["messages"].insert(2, {"role": "user", "content": "[StateFrame]\n" + json.dumps(frame)})
    snap = snapshot(first, second, body=body, current_revisions=(("a.py", first.identity.revision),))
    result = render_assignment(snap, {"first": FULL, "second": DEFERRED})
    rendered = next(m["content"] for m in json.loads(result.wire)["messages"]
                    if m.get("content", "").startswith("[StateFrame]\n"))
    access = json.loads(rendered.split("\n", 1)[1])["state"]["source_access"]
    assert access["included"] is False
    assert access["no_read_call_required_for_this_included_source"] is False


def test_partial_full_slot_cannot_claim_whole_path_source_access():
    slot = source("first", "one\n", first=1)
    body = json.loads(snapshot(slot).fixed_body)
    body["messages"].insert(2, {"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"source_access": {"path": "a.py", "included": True}}})})
    snap = snapshot(slot, body=body, current_revisions=(("a.py", slot.identity.revision),))
    result = render_assignment(snap, {slot.key: FULL})
    rendered = next(m["content"] for m in json.loads(result.wire)["messages"]
                    if m.get("content", "").startswith("[StateFrame]\n"))
    assert json.loads(rendered.split("\n", 1)[1])["state"]["source_access"]["included"] is False


def test_retired_receipts_cannot_inherit_new_same_path_full_coverage():
    slot = replace(source("new", "one\n"), capture_ids=("new-capture",))
    body = json.loads(snapshot(slot).fixed_body)
    fact = {"path": "a.py", "read_revision": slot.identity.revision,
            "current_revision": slot.identity.revision, "lines": {"start": 1, "end": 1},
            "receipt_count": 2, "receipts": [{"receipt_id": "old-retired"},
                                            {"receipt_id": "new-capture"}],
            "body_delivery": "present"}
    body["messages"].insert(2, {"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"read_facts": [fact]}}})})
    snap = snapshot(slot, body=body, current_revisions=(("a.py", slot.identity.revision),))
    result = render_assignment(snap, {slot.key: FULL})
    rendered = next(m["content"] for m in json.loads(result.wire)["messages"]
                    if m.get("content", "").startswith("[StateFrame]\n"))
    fact = json.loads(rendered.split("\n", 1)[1])["state"]["framefit5"]["read_facts"][0]
    # The retired receipt is not claimed; the attached 1-1 range is not denied.
    assert fact["body_delivery"] == "partial"
    assert "lines 1-1 shown; the rest not shown" in fact["delivery_notice"]


def test_witness_or_assignment_mode_flip_on_mixed_candidate_is_rejected():
    """A D notice is bound to the trusted assignment record, not to its prose."""
    full, deferred = source("f", state="DUE"), source("d", "other\n", first=5, ordinal=2, state="RESIDENT")
    result = render_assignment(snapshot(full, deferred), {"f": FULL, "d": DEFERRED}, "plain")
    certificate = result.certificate
    assert verify_certificate(result.wire, certificate)
    witnesses = list(certificate.witnesses)
    index = next(i for i, w in enumerate(witnesses) if w.mode == "DEFERRED")
    flipped = replace(witnesses[index], mode="FULL")
    with pytest.raises(CertificateError):
        verify_certificate(result.wire, replace(certificate, witnesses=(*witnesses[:index], flipped, *witnesses[index + 1:])))
    promoted = tuple((key, "FULL") for key, _mode in certificate.assignment)
    with pytest.raises(CertificateError):
        verify_certificate(result.wire, replace(certificate, assignment=promoted))
    shifted = [replace(w, payload_start=w.payload_start + 1, payload_end=w.payload_end + 1)
               if w.mode == "FULL" else w for w in witnesses]
    with pytest.raises(CertificateError):
        verify_certificate(result.wire, replace(certificate, witnesses=tuple(shifted)))
    # A same-path FULL witness relabelled with a different line range is not accepted either.
    relabelled = [replace(w, receipt_identity=replace(w.receipt_identity, first_line=7, last_line=7))
                  if w.mode == "FULL" else w for w in witnesses]
    with pytest.raises(CertificateError):
        verify_certificate(result.wire, replace(certificate, witnesses=tuple(relabelled)))
