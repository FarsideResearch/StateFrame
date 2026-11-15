"""Review round 2: early selection, reducible floors and shared-carrier bounds."""
import copy
import itertools
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from kernel_working_set_allocator import FULL
from kernel_working_set_prepare import (
    Blocked,
    PreparedCandidate,
    _envelope_variants,
    _mandatory_lower_bound,
    prepare_candidate,
)
from kernel_working_set_proof import verify_certificate
from kernel_working_set_render import propose_slots, render_assignment
from kernel_working_set_snapshot import freeze_preparation
from test_working_set_render import snapshot, source


def frame_body(frame):
    return {"model": "local", "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {
            "framefit5": {"mode": "FRAME_ONLY", **frame}}})},
        {"role": "user", "content": "opaque suffix"}], "max_tokens": 100}


def typed_frame(wire):
    return next(json.loads(row["content"].split("\n", 1)[1])["state"]["framefit5"]
                for row in json.loads(wire)["messages"]
                if row.get("role") == "user" and row.get("content", "").startswith("[StateFrame]\n"))


def head_body(tmp_path):
    from kernel_stateframe_headcarry import _module_head, _panel
    raw = b"IMPORTANT = 42\n"
    (tmp_path / "a.py").write_bytes(raw)
    head = _module_head(tmp_path, "a.py", raw)
    return frame_body({"headcarry": _panel([head], omitted_count=0,
        deduplicated_count=0, byte_budget=6000)})


def test_on_pretrim_preserves_heads_until_allocator_snapshot(tmp_path, monkeypatch):
    from kernel_request_budget import _graceful_trim_optional_messages
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_HEADCARRY", "on")
    body = head_body(tmp_path)
    before = copy.deepcopy(body)
    dropped = _graceful_trim_optional_messages(body, protected_contents=(),
        protected_messages=(), max_request_bytes=1)
    assert body == before, dropped
    frozen = freeze_preparation(body, SimpleNamespace(owner=SimpleNamespace(root=tmp_path)),
                                request_id="proposals", byte_cap=82026)
    slot, = frozen.sources
    assert slot.payload == "IMPORTANT = 42\n"
    assert {mode for mode, _ in propose_slots(frozen)[0].allowed} >= {FULL}


def test_off_pretrim_matches_pinned_base_bytes(tmp_path, monkeypatch):
    from kernel_request_budget import _graceful_trim_optional_messages
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    monkeypatch.setenv("STATEFRAME_HEADCARRY", "on")
    root = Path(__file__).resolve().parents[1]
    original = subprocess.check_output(["git", "show",
        "56af0dc6d6:src/stateframe/engine/kernel_request_budget.py"], cwd=root)
    baseline = ModuleType("_review_budget_base")
    baseline.__file__ = str(root / "src/stateframe/engine/kernel_request_budget.py")
    monkeypatch.setitem(sys.modules, baseline.__name__, baseline)
    # Execute only the pinned local baseline for byte-parity comparison.
    exec(compile(original, baseline.__file__, "exec"), baseline.__dict__)  # noqa: S102
    for cap in (1, 1200, 100_000):
        body = head_body(tmp_path)
        base_body = copy.deepcopy(body)
        kwargs = {"protected_contents": (), "protected_messages": (), "max_request_bytes": cap}
        expected = baseline._graceful_trim_optional_messages(base_body, **kwargs)
        assert _graceful_trim_optional_messages(body, **kwargs) == expected
        assert json.dumps(body).encode() == json.dumps(base_body).encode()


def history_frame(count=24):
    return {"action_ledger": {"reads": [], "edits": [],
        "unclassified_actions": [{"call_id": f"call-{i}", "tool": "shell",
            "text": f"old result {i} " + "x" * 800} for i in range(count)]},
        "owner_protocol": {"admitted_actions": ["read"]}}


def test_singleton_source_does_not_make_reducible_history_mandatory():
    from kernel_stateframe_capacity import ledger_ids, validate_read_ranges
    due = source(text="required\n")
    frame = history_frame()
    snap = snapshot(due, body=frame_body(frame))
    rich = render_assignment(snap, {due.key: FULL})
    snap = replace(snap, byte_cap=len(rich.wire) - 10_000)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), result
    assert result.total_bytes <= snap.byte_cap
    assert dict(result.rendered.certificate.assignment) == {due.key: "FULL"}
    compact = typed_frame(result.wire)
    assert ledger_ids(compact["action_ledger"]) == ledger_ids(frame["action_ledger"])
    validate_read_ranges(compact, json.loads(result.wire)["messages"])
    assert verify_certificate(result.wire, result.rendered.certificate)
    assert snap.fixed_body == snapshot(due, body=frame_body(frame)).fixed_body


def test_existing_journal_coverage_proof_is_unchanged_by_envelope_variants():
    from kernel_stateframe_call_coverage import (
        CoverageSession,
        bind_index,
        context_index,
    )
    from kernel_stateframe_capacity import _fold, ledger_ids, validate_coverage
    from test_coverage_index_base_compat import _MemoryJournal
    journal = _MemoryJournal()
    context = bind_index({}, journal)
    session = CoverageSession(context_index(context), None)
    old = {}
    _fold(old, {"call_ids": ["already-folded"]}, kind="read", coverage=session)
    session.commit()
    frame = history_frame()
    frame["action_ledger"]["history_digest"] = copy.deepcopy(old["history_digest"])
    context["covered_call_ids"] = sorted(ledger_ids(frame["action_ledger"]))
    events = copy.deepcopy(journal.records)
    due = source(text="due\n")
    snap = snapshot(due, body=frame_body(frame), cap=4000)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), result
    compact = typed_frame(result.wire)
    assert compact["action_ledger"]["history_digest"] == old["history_digest"]
    validate_coverage(compact, context)
    assert journal.records == events


def test_rich_envelope_is_retained_when_it_fits():
    due = source(text="due\n")
    snap = snapshot(due, body=frame_body(history_frame()))
    rich = render_assignment(snap, {due.key: FULL})
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate)
    assert result.wire == rich.wire
    assert result.fits_without_degrade


def test_reducible_envelope_without_sources_and_real_mandatory_failure():
    from kernel_working_set_stop import ReasonCode
    snap = snapshot(body=frame_body(history_frame()), cap=2000)
    result = prepare_candidate(snap, measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), result
    failed = prepare_candidate(replace(snap, byte_cap=1))
    assert isinstance(failed, Blocked)
    assert failed.reason is ReasonCode.FLOOR_INFEASIBLE
    failed.proof.validate()
    assert len(failed.proof.mandatory_wire) <= result.total_bytes


@pytest.mark.parametrize("with_due", [False, True])
def test_floor_proof_uses_the_smallest_rendered_envelope(with_due):
    from kernel_working_set_stop import ReasonCode
    sources = (source(),) if with_due else ()
    snap = snapshot(*sources, body=frame_body(history_frame()))
    assignment = {slot.key: FULL for slot in sources}
    smallest = min((render_assignment(variant, assignment) for variant in _envelope_variants(snap)),
                   key=lambda candidate: len(candidate.wire))
    result = prepare_candidate(replace(snap, byte_cap=len(smallest.wire)),
                               measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), result
    assert result.wire == smallest.wire
    failed = prepare_candidate(replace(snap, byte_cap=len(smallest.wire) - 1))
    assert isinstance(failed, Blocked)
    assert failed.reason is ReasonCode.FLOOR_INFEASIBLE
    assert failed.proof.mandatory_wire == smallest.wire


@pytest.mark.parametrize("recovery", [False, True])
def test_compact_site_routes_must_match_the_frozen_recovery_surface(recovery):
    from kernel_open_site_capacity import LOCATION_MODE
    a, b = source(text="one\n"), source("b", text="two\n", first=2, ordinal=2)
    sites = [{"source": "test", "path": slot.identity.path, "line": slot.identity.first_line,
        "literal": "", "revision": slot.identity.revision,
        "lines": {"start": slot.identity.first_line, "end": slot.identity.last_line},
        "content_sha256": slot.identity.payload_sha256, "current_bytes": slot.payload}
        for slot in (a, b)]
    snap = snapshot(a, b, body=frame_body({"next_site": sites[0], "open_sites": sites}),
        recovery_allowed=recovery, recovery_schema=json.dumps({"type": "object",
            "properties": {"path": {"type": "string"}, "cursor": {"type": "string"}},
            "required": ["path", "cursor"], "additionalProperties": False}).encode())
    for variant in _envelope_variants(snap):
        assert typed_frame(variant.fixed_body).get("site_inventory_encoding") != LOCATION_MODE


def test_envelope_variants_leave_opaque_suffix_and_exchange_byte_exact():
    body = frame_body(history_frame())
    opaque = copy.deepcopy(body["messages"][2])
    body["messages"].append(opaque)
    body["messages"].extend([{"role": "assistant", "content": "native exchange"},
                             {"role": "tool", "content": "opaque tool result"}])
    snap = snapshot(body=body, insertion_index=3)
    for variant in _envelope_variants(snap):
        assert json.loads(variant.fixed_body)["messages"][3:] == body["messages"][3:]


@pytest.mark.parametrize("initial", ["", "old"])
def test_compact_envelope_certificate_survives_tokenizer_binding(initial):
    snap = snapshot(source(), body=frame_body(history_frame()), cap=4000,
                    tokenizer_identity=initial)
    from kernel_working_set_prepare import Measurement
    result = prepare_candidate(snap, measure=lambda wire: Measurement(1, 0, "test", "new"))
    assert isinstance(result, PreparedCandidate), result
    assert result.rendered.certificate.snapshot.tokenizer_identity == "new"
    assert verify_certificate(result.wire, result.rendered.certificate)


@pytest.mark.parametrize("count", [1, 3])
def test_site_inventory_has_priced_compact_representation(count):
    from kernel_frame_optional_allocation import singleton_alias_candidate
    from kernel_open_site_capacity import location_inventory_candidate
    due = source(text="a" * 1800 + "\n")
    site = {"source": "test", "path": due.identity.path, "line": 1,
        "literal": "a", "revision": due.identity.revision,
        "lines": {"start": 1, "end": 1},
        "content_sha256": due.identity.payload_sha256, "current_bytes": due.payload}
    sites = [copy.deepcopy(site) for _ in range(count)]
    frame = {"next_site": site, "open_sites": sites}
    compact = (singleton_alias_candidate(frame) if count == 1
               else location_inventory_candidate(frame))
    assert compact is not None
    snap = snapshot(due, body=frame_body(frame), current_revisions=((due.identity.path,
        due.identity.revision),))
    rich = render_assignment(snap, {due.key: FULL})
    expected = render_assignment(replace(snap, fixed_body=json.dumps(frame_body(compact)).encode()),
                                 {due.key: FULL})
    assert len(expected.wire) < len(rich.wire)
    result = prepare_candidate(replace(snap, byte_cap=len(expected.wire)),
                               measure=lambda wire: (1, 0, "test"))
    assert isinstance(result, PreparedCandidate), result
    assert result.total_bytes <= len(expected.wire)
    assert verify_certificate(result.wire, result.rendered.certificate)


def test_mandatory_bound_shares_native_page_with_advisory_suffix():
    due = source(text="captured source line " * 240 + "\n")
    body = json.loads(snapshot(due).fixed_body)
    body["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [{"id": "read-1",
            "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "read-1",
         "content": due.page + "\n[read] current requested content is in this result."}])
    snap = snapshot(due, body=body)
    valid = render_assignment(snap, {due.key: FULL})
    assert verify_certificate(valid.wire, valid.certificate)
    bound = _mandatory_lower_bound(snap, propose_slots(snap))
    assert len(bound) <= len(valid.wire), (len(bound), len(valid.wire))


def test_mandatory_bound_is_below_every_generated_legal_candidate():
    """Finite property check: enumerate every legal mode for varied frozen inputs."""
    rng = random.Random(904)
    for case in range(16):
        text = rng.choice(["λ\\\"\t", "plain", "🛰"]) * rng.randrange(800, 1100) + "\n"
        due = source(text=text)
        optional = source("optional", text="other\n", first=2,
                          state="RESIDENT", ordinal=2)
        duplicate = replace(due, key="duplicate", obligation_ids=("duplicate",),
                            admission_ordinal=3)
        body = frame_body(history_frame(2))
        frame = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
        frame["state"]["framefit5"]["owner_protocol"]["continuation_limits"] = (
            "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2")
        body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps(frame)
        if case % 4:
            annotation = "\n[read] advisory" if case % 2 else ""
            body["messages"].extend([
                {"role": "assistant", "tool_calls": [{"id": "r", "type": "function",
                    "function": {"name": "read", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "r", "content": due.page + annotation}])
        body["messages"].append({"role": "user", "content": "[WIRE VOLATILE TAIL]"})
        snap = snapshot(due, optional, duplicate, body=body, layouts=("plain", "final_cache"))
        slots = propose_slots(snap)
        bound = _mandatory_lower_bound(snap, slots)
        for choices in itertools.product(*(slot.allowed for slot in slots)):
            modes = {slot.key: mode for slot, (mode, _) in zip(slots, choices)}
            for variant in _envelope_variants(snap):
                for layout in snap.layouts:
                    valid = render_assignment(variant, modes, layout)
                    assert verify_certificate(valid.wire, valid.certificate)
                    assert len(bound) <= len(valid.wire), (case, layout, len(bound), len(valid.wire))
