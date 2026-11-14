"""Accepted historical source exposure survives residency eviction for native edits."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from kernel_continuation_evidence import HostEvidence
from kernel_continuation_state import ContinuationError
from kernel_native_effect_chain import OrderedSources, revision
from kernel_source_delivery import deliver_source
from kernel_working_set_state import record_capture, record_response, record_submission, state_for
from kernel_working_set_windows import render_frozen_span
from test_working_set_state import capture


PATH = "source.py"
SOURCE = "first = 1\nvalue = 2\nlast = 3\n"
EDIT = {"path": PATH, "old_str": "value = 2", "new_str": "value = 9"}


class Journal:
    def __init__(self):
        self.evidence = HostEvidence("native-history", native_prefix=True)
        self.events = []

    def emit(self, kind, **fields):
        event = {"kind": kind, "run_id": "native-history",
                 "event_id": f"event-{len(self.events) + 1}",
                 "sequence": len(self.events) + 1, **fields}
        self.events.append(event)
        self.evidence.observe(event)
        return event


def _wire(rt, request, messages, modes, disposition):
    body = {"messages": messages, "tools": []}
    wire = json.dumps(body)
    checksum = hashlib.sha256(wire.encode()).hexdigest()
    prepared = rt.journal.emit("request_prepared", request_id=request,
        snapshot_id=request, messages=messages, overflow=None, body_artifact="prewire_draft")
    rt.journal.emit("request_final_wire_prepared", request_id=request,
        snapshot_id=request, transport_attempt=1, producer="kernel_measurement",
        event_schema="stateframe-request-final-wire-prepared/1",
        body_artifact="final_wire_pre_dispatch", provider_request=True, dispatch=False,
        payload_sha256="sha256:" + checksum, **body)
    sent = rt.journal.emit("request_submitted", request_id=request,
        snapshot_id=request, transport_attempt=1, payload_utf8=wire,
        payload_hash="sha256:" + checksum, payload=body)
    record_submission(rt, request_id=request, body_sha256=checksum, modes=modes)
    received = rt.journal.emit("response_received", request_id=request,
        transport_attempt_id=sent["event_id"], response={})
    record_response(rt, request_id=request, response_id=received["event_id"],
        body_sha256=checksum, disposition=disposition)
    return prepared, body


def _evicted_sources(monkeypatch, *, exposed=SOURCE, disposition="accepted", first=2, last=2,
                     carrier="window"):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    rt = SimpleNamespace(journal=Journal())
    lines = exposed.splitlines(keepends=True)
    payload = "".join(lines[first - 1:last])
    read = capture(payload=payload)
    read = replace(read, identity=replace(read.identity, revision=revision(exposed),
        first_line=first, last_line=last))
    record_capture(rt, read)
    obligation, = state_for(rt).active_obligations()
    page, _ = render_frozen_span({"path": PATH, "revision": revision(exposed),
        "total_lines": len(lines)}, {"start_line": first, "end_line": last}, payload)
    if carrier == "working":
        page = f"[Working source: {PATH}]\n{exposed}"
    elif carrier == "read":
        page = deliver_source(exposed.encode(), PATH, 0,
            args={"path": PATH, "start_line": first, "end_line": last}).text
    _wire(rt, "exposure", [{"role": "tool" if carrier == "read" else "user", "content": page}],
          {obligation.obligation_id: "F"}, disposition)
    if disposition == "accepted":
        assert obligation.state == "RESIDENT"
        _wire(rt, "eviction", [{"role": "user", "content": "Task: edit"}],
              {obligation.obligation_id: "O"}, "accepted")
        assert not state_for(rt).active_obligations()
    rt.journal.emit("request_source_closure", producer="kernel_request_source_window",
        dispatch=False, closure={"revisions": {PATH: revision(SOURCE)},
                                 "source_contents": {PATH: SOURCE}})
    prepared = rt.journal.emit("request_prepared", request_id="batch", snapshot_id="batch")
    return rt, OrderedSources([], prepared,
        {"messages": [{"role": "user", "content": "Task: edit"}]},
        {"product_files": {PATH: revision(SOURCE)}}, final_delivery=True,
        source_evidence=rt.journal.evidence)


@pytest.mark.parametrize("carrier", ["window", "working", "read"])
def test_accepted_exposure_then_eviction_authorizes_native_batch_edit(monkeypatch, carrier):
    _, sources = _evicted_sources(monkeypatch, carrier=carrier)
    assert sources.apply("str_replace", EDIT, 1, pending=True) == PATH
    assert sources.texts[PATH] == SOURCE.replace("value = 2", "value = 9")


@pytest.mark.parametrize("disposition", ["discarded", "failed"])
def test_unaccepted_exposure_cannot_authorize_native_batch_edit(monkeypatch, disposition):
    _, sources = _evicted_sources(monkeypatch, disposition=disposition)
    with pytest.raises(ContinuationError, match="anchor was not delivered"):
        sources.apply("str_replace", EDIT, 1, pending=True)


@pytest.mark.parametrize("carrier", ["window", "working", "read"])
def test_stale_exposure_cannot_authorize_native_batch_edit(monkeypatch, carrier):
    _, sources = _evicted_sources(monkeypatch, exposed=SOURCE.replace("last = 3", "last = 4"),
                                  carrier=carrier)
    with pytest.raises(ContinuationError, match="anchor was not delivered"):
        sources.apply("str_replace", EDIT, 1, pending=True)


def test_unexposed_preimage_cannot_borrow_another_range(monkeypatch):
    _, sources = _evicted_sources(monkeypatch, first=1, last=1)
    with pytest.raises(ContinuationError, match="anchor was not delivered"):
        sources.apply("str_replace", EDIT, 1, pending=True)


@pytest.mark.parametrize("corruption", ["missing_acceptance", "body_hash", "response_identity",
    "missing_final_wire", "final_wire", "submitted_bytes", "late_acceptance"])
def test_historical_exposure_requires_verified_acceptance_lineage(monkeypatch, corruption):
    rt, sources = _evicted_sources(monkeypatch)
    events = json.loads(json.dumps(rt.journal.events))
    receipt = next(e for e in events if e["kind"] == "working_set_response"
                   and e["request_id"] == "exposure")
    final = next(e for e in events if e["kind"] == "request_final_wire_prepared"
                 and e["request_id"] == "exposure")
    if corruption == "missing_acceptance":
        events.remove(receipt)
    elif corruption == "body_hash":
        receipt["body_sha256"] = "0" * 64
    elif corruption == "response_identity":
        receipt["request_id"] = "foreign"
    elif corruption == "missing_final_wire":
        events.remove(final)
    elif corruption == "final_wire":
        final["messages"] = []
    elif corruption == "submitted_bytes":
        sent = next(e for e in events if e["kind"] == "request_submitted"
                    and e["request_id"] == "exposure")
        sent["payload_utf8"] += " "
    else:
        receipt["sequence"] = sources.prepared["sequence"] + 1
    evidence = HostEvidence("native-history", native_prefix=True)
    for event in events:
        evidence.observe(event)
    sources.source_evidence = evidence
    with pytest.raises(ContinuationError, match="anchor was not delivered"):
        sources.apply("str_replace", EDIT, 1, pending=True)


def test_historical_exposure_preserves_uniqueness_check(monkeypatch):
    _, sources = _evicted_sources(monkeypatch)
    sources.apply("str_replace", {"path": PATH, "old_str": "last = 3", "new_str": "value = 2"},
                  0, pending=False)
    with pytest.raises(ContinuationError, match="old_str is no longer unique"):
        sources.apply("str_replace", EDIT, 1, pending=True)


def test_historical_exposure_preserves_effect_overlap_check(monkeypatch):
    _, sources = _evicted_sources(monkeypatch)
    sources.apply("str_replace", {**EDIT, "new_str": "value = 2 # changed"}, 0, pending=False)
    with pytest.raises(ContinuationError, match="overlaps call 1"):
        sources.apply("str_replace", EDIT, 1, pending=True)


def test_historical_exposure_is_disabled_with_allocator_off(monkeypatch):
    _, sources = _evicted_sources(monkeypatch)
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    with pytest.raises(ContinuationError, match="anchor was not delivered"):
        sources.apply("str_replace", EDIT, 1, pending=True)


def test_off_provider_bytes_match_g6_base(tmp_path, tmp_path_factory):
    from test_working_set_astra_off_parity import test_off_provider_bytes_match_cutover_base

    test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, "936a59be20")
