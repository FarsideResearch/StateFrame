"""First-request notice pressure, durable diagnostics and bounded read admission."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
from kernel_working_set_proof import CertificateError, verify_certificate
from kernel_working_set_render import render_assignment
from kernel_working_set_allocator import DEFERRED, FULL
from test_working_set_prepare import source, snapshot


def pressure_snapshot():
    slots = tuple(source(f"docs/guide/topic_{i:03}", state="CLOSURE", ordinal=i + 1)
                  for i in range(98))
    snap = snapshot(*slots, byte_cap=82_026)
    body = json.loads(snap.fixed_body)
    body["messages"][0]["content"] = "policy " * 6400
    body["max_tokens"] = 24_000
    return replace(snap, fixed_body=json.dumps(body).encode(), frame_limit=40_000,
                   context_limit=65_536, output_reserve=4096, safety_reserve=4096)


def test_first_request_optional_notice_pressure():
    result = prepare_candidate(pressure_snapshot())
    assert isinstance(result, PreparedCandidate), result
    assert result.total_bytes <= 82_026
    assert result.trials and not result.trials[-1].violated_limits
    assert len(result.rendered.certificate.witnesses) == 98
    assert verify_certificate(result.wire, result.rendered.certificate)
    print(f"floor={result.floor_bytes} selected={result.total_bytes}")


def test_first_request_absent_plan_units_pressure():
    snap = pressure_snapshot()
    snap = replace(snap, sources=tuple(replace(s, plan_only=True) for s in snap.sources))
    result = prepare_candidate(snap)
    assert isinstance(result, PreparedCandidate), result
    assert result.floor_bytes < 55_000 and result.total_bytes <= 82_026
    certificate = result.rendered.certificate
    assert certificate.deferred_plan_witness is not None
    assert len(certificate.deferred_plan_witness.slot_ids) + len(certificate.witnesses) == 98
    assert verify_certificate(result.wire, certificate)
    print(f"aggregated floor={result.floor_bytes} selected={result.total_bytes}")


@pytest.mark.parametrize("route", ["range", "max_lines", "cursor"])
def test_deferred_notice_omits_only_metadata_encoded_in_validated_call(route):
    from kernel_working_set_render import _notice
    from test_working_set_recovery import schema
    slot = source("a")
    snap = snapshot(slot)
    if route == "max_lines":
        snap = replace(snap, recovery_schema=schema("path", "start_line", "max_lines", "max_chars"))
    elif route == "cursor":
        cursor = "read1:" + slot.identity.revision.removeprefix("sha256:") + ":1:0"
        slot = replace(slot, recovery_arguments=json.dumps({"path": "a.py", "cursor": cursor}).encode())
        snap = replace(snap, sources=(slot,), recovery_schema=schema("path", "cursor", "max_lines", "max_chars"))
    notice = _notice(snap, slot)
    metadata = notice.splitlines()[1]
    assert f"digest={slot.identity.payload_sha256}" in metadata
    assert ("revision=" in metadata) == (route != "range")
    assert ("lines=" in metadata) == (route != "range")
    assert 'path="a.py"' in notice and "Retrieve current bytes with: read(" in notice


def plan_snapshot():
    from kernel_working_set_windows import frozen_plan_windows
    from kernel_request_source_window import source_revision
    from kernel_working_set_recovery import _schema
    contents = {f"docs/topic_{i}.py": "a\nb\nc\n" for i in range(8)}
    revisions = {path: source_revision(text) for path, text in contents.items()}
    units = [{"path": path, "start_line": n, "end_line": n, "role": "context",
              "unit_handle": f"{path}:{n}"}
             for path in contents for n in (1, 3)]
    plan = {"units": units, "paths": list(contents), "revisions": revisions, "source_contents": contents,
            "workspace_authorization": {"scope": "review", "generation": "0"},
            "deferred_paths": list(contents),
            "delivery": {path: {"status": "PARTIAL", "retrieval_route": {
                "status": "EXECUTABLE_BOUNDED_READ", "tool": "read", "path": path,
                "revision": revisions[path],
                "unit_handles": [u["unit_handle"] for u in units if u["path"] == path]}}
                for path in contents}}
    slots, _ = frozen_plan_windows(SimpleNamespace(_request_source_closure=plan), (), {})
    snap = snapshot(*slots, current_revisions=tuple(revisions.items()))
    body = json.loads(snap.fixed_body)
    body["tools"] = [{"type": "function", "function": {
        "name": "read", "parameters": _schema(snap)}}]
    return replace(snap, fixed_body=json.dumps(body).encode()), plan


@pytest.mark.parametrize("layout", ["plain", "final_cache"])
def test_absent_plan_units_share_one_strict_witness_and_pass_closure(layout):
    from kernel_request_source_window import source_delivery_satisfied, verify_final_source_delivery
    from test_final_cache_layout import _body
    snap, plan = plan_snapshot()
    body = _body(exchange=True)
    del body["messages"][3]  # The fixture's unrelated placeholder source.
    body["tools"] = json.loads(snap.fixed_body)["tools"]
    snap = replace(snap, fixed_body=json.dumps(body).encode(), insertion_index=3,
                   layouts=("plain", "final_cache"))
    rendered = render_assignment(snap, {s.key: DEFERRED for s in snap.slots}, layout)
    certificate = rendered.certificate
    assert verify_certificate(rendered.wire, certificate)
    assert not certificate.witnesses
    aggregate = certificate.deferred_plan_witness
    assert aggregate.slot_ids == tuple(s.key for s in snap.slots)
    messages = json.loads(rendered.wire)["messages"]
    notices = [m["content"] for m in messages if "[Deferred required source paths]" in m.get("content", "")]
    assert len(notices) == 1 and "[Source page deferred]" not in rendered.wire.decode()
    receipt = verify_final_source_delivery(plan, messages, payload=rendered.wire)
    assert source_delivery_satisfied(receipt), receipt
    assert receipt["status"] == "DEFERRED" and not receipt["proof_failures"]
    with pytest.raises(CertificateError):
        verify_certificate(rendered.wire, replace(certificate, deferred_plan_witness=replace(
            aggregate, slot_ids=aggregate.slot_ids[:-1])))
    with pytest.raises(CertificateError):
        verify_certificate(rendered.wire, replace(certificate, deferred_plan_witness=None))


def test_freeze_aggregates_only_plan_units_absent_from_current_panels():
    from kernel_working_set_snapshot import freeze_preparation
    snap, plan = plan_snapshot()
    body = json.loads(snap.fixed_body)
    body["messages"][2:2] = [
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{"owner_protocol":{"admitted_actions":["read"]}}}}'},
        {"role": "user", "content": snap.slots[0].page}]
    frozen = freeze_preparation(body, SimpleNamespace(_request_source_closure=plan),
                                request_id="freeze", byte_cap=100_000)
    assert sum(s.plan_only for s in frozen.sources) == len(snap.slots) - 1
    rendered = render_assignment(frozen, {s.key: DEFERRED for s in frozen.slots})
    assert len(rendered.certificate.witnesses) == 1
    assert len(rendered.certificate.deferred_plan_witness.slot_ids) == len(snap.slots) - 1
    assert verify_certificate(rendered.wire, rendered.certificate)


def test_plan_aggregation_never_absorbs_due_or_previously_delivered_pages():
    snap, _ = plan_snapshot()
    due = replace(snap.slots[0], state="DUE")
    captured = replace(snap.slots[1], capture_ids=("capture",), state="RESIDENT")
    snap = replace(snap, sources=(due, captured, *snap.slots[2:]))
    rendered = render_assignment(snap, {s.key: FULL if s == due else DEFERRED for s in snap.slots})
    assert [w.slot_id for w in rendered.certificate.witnesses] == [due.key, captured.key]
    assert verify_certificate(rendered.wire, rendered.certificate)
    assert "[Source page deferred]" in rendered.wire.decode()


def test_blocked_preserves_policy_floor_diagnostics(tmp_path, monkeypatch):
    from kernel_working_set_live import _blocked
    from kernel_working_set_stop import ReasonCode, Suspension, WorkingSetSuspended
    from test_read_receipts_admission import runtime
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    rt = runtime(tmp_path)
    result = Blocked(ReasonCode.BYTE_POLICY_DOMAIN_EXHAUSTED, 92_497, 92_497, 82_026,
                     (("fixed_and_layout", 45_000), ("deferred_notices", 47_497)),
                     detail="policy minimum exceeds cap without an impossibility proof")
    with pytest.raises(WorkingSetSuspended) as raised:
        _blocked(rt, snapshot(), result)
    document = raised.value.suspension.document()
    for key in ("floor_bytes", "cap", "detail"):
        assert document[key] == getattr(result, key)
    assert document["section_breakdown"] == dict(result.section_breakdown)
    assert Suspension.from_document(json.loads(json.dumps(document))).document() == document
    event, = [e for e in rt.journal.events if e["kind"] == "working_set_blocked"]
    assert event["suspension"] == document
    assert document["proof"] is None


@pytest.mark.parametrize("function", ["ask", "ask_tools"])
def test_suspension_has_exactly_one_preparation_attempt(monkeypatch, function):
    from kernel_measurement import ACTIVE_REQUEST
    from probes import react_repair as react
    from kernel_working_set_stop import WorkingSetSuspended
    from test_working_set_stop import suspension
    attempts = []
    signal = WorkingSetSuspended(suspension())
    def prepare(*args, **kwargs):
        attempts.append(args)
        raise signal
    capture = SimpleNamespace(send=prepare, request_id="blocked", snapshot_id="snapshot", step=1,
                              journal=SimpleNamespace(emit=lambda *a, **k: None))
    monkeypatch.setattr("kernel_request_kind.require_capture", lambda *a: None)
    monkeypatch.setattr("sampler_config.journal_sent", lambda *a, **k: None)
    monkeypatch.setattr(react.time, "sleep", lambda *a: None)
    monkeypatch.setattr("kernel_measurement.per_request_timeout_seconds", lambda: None)
    token = ACTIVE_REQUEST.set(capture)
    try:
        with pytest.raises(WorkingSetSuspended) as raised:
            kwargs = {"tools": []} if function == "ask_tools" else {}
            getattr(react, function)([{"role": "user", "content": "task"}], **kwargs)
        assert raised.value is signal
        assert len(attempts) == 1
    finally:
        ACTIVE_REQUEST.reset(token)


def admission_case(lines, *, cap=100_000):
    from kernel_source_delivery import deliver_source, normalize_read_request
    from kernel_working_set_live import read_admission_floor
    import kernel_read_admission as admission
    raw = "".join(lines).encode()
    request = normalize_read_request({"path": "source.py", "max_lines": len(lines), "max_chars": 64000})
    page = deliver_source(raw, "source.py", 0, args=request)
    snap = snapshot(byte_cap=cap)
    floor_bytes = len(read_admission_floor(snap).wire)
    state = admission._RequestState(1, cap - floor_bytes, snapshot=snap, floor_bytes=floor_bytes)
    rt = SimpleNamespace(_frame_budget_request_count=1, _read_admission_state=state)
    return rt, raw, request, page


@pytest.mark.parametrize("window_lines", [0, 45, 95],
                         ids=["no-plan", "feasible-plan", "over-cap-plan"])
def test_120_line_admission_uses_at_most_eight_floors_and_memoizes(monkeypatch, window_lines):
    import kernel_working_set_live as live
    import kernel_read_admission as admission
    rt, raw, request, page = admission_case([f"{i:03} " + "x" * 100 + "\n" for i in range(120)], cap=8000)
    original = live.read_admission_floor
    state = rt._read_admission_state
    if window_lines:
        from kernel_source_delivery import deliver_source
        window = deliver_source(raw, "source.py", 0, args=replace(request, max_lines=window_lines))
        slot = live.read_admission_slot(window, path="source.py", arguments=request.arguments, ordinal=1)
        prefix = "[Current source: source.py]\n"
        slot = replace(slot, key="plan", state="CLOSURE", plan_only=True,
                       page=prefix + window.content, payload_start=len(prefix.encode()))
        state.snapshot = replace(state.snapshot, sources=(slot,))
        state.floor_bytes = len(original(state.snapshot).wire)
        state.allowance = state.snapshot.byte_cap - state.floor_bytes
    calls = []
    def floor(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(live, "read_admission_floor", floor)
    result = admission.admit(rt, raw=raw, display="source.py", start=0, request=request,
                             source=page, visible_messages=None)
    # Promotion makes the whole existing plan range mandatory. The 95-line
    # range alone exceeds 8 KB, so even a one-line fresh page must be refused.
    if window_lines == 95:
        assert result.result_class == admission.REFUSED_ADMISSION
        assert not result.content
    else:
        assert result.result_class == "OK" and len(result.content.splitlines()) < 120
    assert result.metadata["next_cursor"] in result.text
    # Eight search floors plus one exact boundary probe: frozen carrier-boundary
    # prefixes are priced by cost() without a carrier-equality precheck.
    assert len(calls) <= 9
    n = len(result.content.splitlines())
    from kernel_source_delivery import deliver_source
    candidate = deliver_source(raw, "source.py", 0, args=replace(request, max_lines=n + 1))
    ordinal = max((s.admission_ordinal for s in state.snapshot.sources), default=0) + 1
    slot = live.read_admission_slot(candidate, path="source.py", arguments=request.arguments, ordinal=ordinal)
    assert len(original(state.snapshot, ((slot, None),)).wire) > state.snapshot.byte_cap
    print(f"admitted_lines={n} floor_computations={len(calls)}")
    # Revisit the same frozen snapshot and admitted ranges within this request.
    count = len(calls)
    state.pages.clear()
    state.pending_bytes = 0
    state.closed = False
    again = admission.admit(rt, raw=raw, display="source.py", start=0, request=request,
                            source=page, visible_messages=None)
    assert again == result and len(calls) == count


@pytest.mark.parametrize("seed", range(12))
def test_binary_admission_matches_linear_largest_prefix(seed):
    import random
    import kernel_read_admission as admission
    from kernel_source_delivery import deliver_source
    from kernel_working_set_live import read_admission_floor, read_admission_slot
    rng = random.Random(seed)
    lines = [(rng.choice(['x', 'λ', '😀', '"\\']) * rng.randint(1, 80)) + "\n"
             for _ in range(rng.randint(2, 15))]
    rt, raw, request, page = admission_case(lines)
    state = rt._read_admission_state
    floors = []
    for n in range(1, len(lines) + 1):
        candidate = page if n == len(lines) else deliver_source(raw, "source.py", 0, args=replace(request, max_lines=n))
        slot = read_admission_slot(candidate, path="source.py", arguments=request.arguments, ordinal=1)
        floors.append(len(read_admission_floor(state.snapshot, ((slot, None),)).wire))
    # Check every threshold, including refusal, full delivery, and exact fits.
    for cap in (state.floor_bytes, *(n + d for n in floors for d in (-1, 0, 1))):
        state.snapshot = replace(state.snapshot, byte_cap=cap)
        state.allowance = max(0, cap - state.floor_bytes)
        state.pages.clear()
        state.pending_bytes = 0
        state.closed = False
        expected = max((n for n, size in enumerate(floors, 1) if size <= cap), default=0)
        result = admission.admit(rt, raw=raw, display="source.py", start=0, request=request,
                                 source=page, visible_messages=None)
        assert (len(result.content.splitlines()) if result.result_class == "OK" else 0) == expected


@pytest.mark.parametrize("shared_count", [2, 4, 7])
@pytest.mark.parametrize("display", ["source.py", "./source.py"])
def test_binary_admission_preserves_exact_carrier_sharing_dips(shared_count, display, tmp_path):
    import kernel_read_admission as admission
    from kernel_source_delivery import deliver_source
    from kernel_working_set_live import read_admission_floor, read_admission_slot
    rt, raw, request, page = admission_case(["x\n"] * 8)
    rt.owner = SimpleNamespace(root=tmp_path)
    page = deliver_source(raw, display, 0, args=request)
    state = rt._read_admission_state
    existing = deliver_source(raw, display, 0, args=replace(request, max_lines=shared_count))
    slot = read_admission_slot(existing, path="source.py", arguments=request.arguments, ordinal=1)
    prior = replace(slot, key="prior", obligation_ids=("prior",), state="RESIDENT")
    state.snapshot = replace(state.snapshot, sources=(prior,))
    state.floor_bytes = len(read_admission_floor(state.snapshot).wire)
    sizes = []
    for n in range(1, 9):
        candidate = deliver_source(raw, display, 0, args=replace(request, max_lines=n))
        proposal = read_admission_slot(candidate, path="source.py", arguments=request.arguments, ordinal=2)
        sizes.append(len(read_admission_floor(state.snapshot, ((proposal, None),)).wire))
    for cap in sorted({size + d for size in sizes for d in (-1, 0, 1)}):
        state.snapshot = replace(state.snapshot, byte_cap=cap)
        state.allowance = max(0, cap - state.floor_bytes)
        state.pages.clear()
        state.pending_bytes = 0
        state.closed = False
        expected = max((n for n, size in enumerate(sizes, 1) if size <= cap), default=0)
        result = admission.admit(rt, raw=raw, display=display, start=0, request=request,
                                 source=page, visible_messages=None)
        assert (len(result.content.splitlines()) if result.result_class == "OK" else 0) == expected
