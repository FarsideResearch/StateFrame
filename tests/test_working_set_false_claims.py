"""Coverage healing must precede the allocator's exact-wire certificate."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from kernel_working_set_proof import verify_certificate


@pytest.mark.parametrize("copies", [0, 1, 2], ids=["empty", "rewrite", "deduplicate"])
def test_on_heals_stale_suffix_before_certification_and_seal(tmp_path, monkeypatch, copies):
    import kernel_working_set_live as live
    from test_read_admission_qualification import seal_case

    original = live.prepare_live
    prepared = []
    stale = "[Source coverage]\nghost.py (99 lines, unchanged since last read): all 99 lines are shown below."

    def inject(capture, runtime, body, **kwargs):
        # This late host block survives freeze_preparation's opaque suffix.
        content = stale if copies else "[Source coverage]\nobsolete row"
        suffix = [{"role": "user", "content": content} for _ in range(max(1, copies))]
        suffix.append({"role": "user", "content": "Keep this opaque user instruction unchanged: λ"})
        index = next(i for i, row in enumerate(body["messages"])
                     if row.get("role") == "assistant")
        body["messages"][index:index] = suffix
        capture._framefit5_newest_user_suffix = tuple(suffix)
        result = original(capture, runtime, body, **kwargs)
        prepared.append(result[1])
        return result

    monkeypatch.setattr(live, "prepare_live", inject)
    record, capture, _ = seal_case(tmp_path, monkeypatch, "K1")
    certified, = prepared
    assert record.wire == certified.wire == json.dumps(record.body).encode()
    assert verify_certificate(record.wire, capture._working_set_certificate)
    assert stale not in [row.get("content") for row in record.body["messages"]]
    assert any("ghost.py: not shown;" in str(row.get("content", ""))
               for row in record.body["messages"]) == bool(copies)
    assert not any("giant.py: not shown;" in str(row.get("content", ""))
                   for row in record.body["messages"])
    assert sum(row.get("content") == "Keep this opaque user instruction unchanged: λ"
               for row in record.body["messages"]) == 1


def test_final_wire_check_is_explicitly_read_only(tmp_path, monkeypatch):
    import kernel_frame_source_snapshot as coverage
    from test_read_admission_qualification import seal_case

    checks = []
    original = coverage.verify_source_coverage_claims

    def verify(messages, **kwargs):
        before = copy.deepcopy(messages)
        checks.append(kwargs.get("strict"))
        result = original(messages, **kwargs)
        assert messages == before
        return result

    monkeypatch.setattr(coverage, "verify_source_coverage_claims", verify)
    seal_case(tmp_path, monkeypatch, "K1")
    assert checks == [True]


def test_production_heal_then_strict_check_is_idempotent():
    from kernel_frame_source_snapshot import verify_source_coverage_claims, SourceCoverageClaimError

    messages = [{"role": "user", "content":
                 "[Source coverage]\nghost.py (1000000000000 lines, unverified): "
                 "all 1000000000000 lines are shown below."}]
    original = copy.deepcopy(messages)
    with pytest.raises(SourceCoverageClaimError):
        verify_source_coverage_claims(messages, strict=True)
    assert messages == original
    from types import SimpleNamespace
    from kernel_continuation_restart import SAFE_TAIL, resume_event_rejection_reason
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    assert not verify_source_coverage_claims(messages, strict=False, journal=journal,
                                             request_id="healed")
    assert len(events) == 1
    kind, fields = events[0]
    assert kind == "context_claim_reconciled" and fields["healed"] is True
    assert kind in SAFE_TAIL and resume_event_rejection_reason(kind) is None
    healed = copy.deepcopy(messages)
    assert verify_source_coverage_claims(messages, strict=True)
    assert messages == healed
    assert "ghost.py: not shown;" in messages[0]["content"]


@pytest.mark.parametrize("layout", ["plain", "final_cache"])
@pytest.mark.parametrize("contents", [
    ["[Source coverage]\nunrecognized obsolete row"],
    ["[Source coverage]\nghost.py (1 line, unverified): the single line is shown below."] * 2,
])
def test_heal_removals_preserve_fixed_tool_carrier_coordinates(layout, contents):
    from test_final_cache_layout import _body
    from test_working_set_render import source, snapshot
    from kernel_working_set_allocator import FULL
    from kernel_working_set_render import render_assignment

    slot = source()
    body = _body(exchange=True)
    state = json.loads(body["messages"][2]["content"].split("\n", 1)[1])
    state["state"]["framefit5"]["mode"] = "FRAME_ONLY"
    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps(state)
    tool = next(row for row in body["messages"] if row.get("role") == "tool")
    tool["content"] = slot.page
    # These advisory rows precede the fixed tool carrier; removal shifts it.
    body["messages"][3:3] = [{"role": "user", "content": text} for text in contents]
    original = copy.deepcopy(body)
    frozen = snapshot(slot, body=body, insertion_index=3, layouts=("plain", "final_cache"))
    rendered = render_assignment(frozen, {slot.key: FULL}, layout)
    assert verify_certificate(rendered.wire, rendered.certificate)
    witness, = rendered.certificate.witnesses
    wire_body = json.loads(rendered.wire)
    assert wire_body["messages"][witness.message_index]["content"] == slot.page
    assert wire_body["messages"][witness.message_index]["role"] == "tool"
    assert body == original
    assert b"unrecognized obsolete row" not in rendered.wire
    assert rendered.wire.count(b"ghost.py: not shown;") <= 1


@pytest.mark.parametrize("clipped", [False, True])
def test_heal_uses_window_body_and_refreshes_content_digest(tmp_path, clipped):
    from kernel_frame_source_snapshot import (
        _current_window, reconcile_source_coverage,
        verify_source_coverage_claims, SourceCoverageClaimError,
    )
    from kernel_request_source_window import source_revision

    source = "one\ntwo\n"
    (tmp_path / "a.py").write_text(source)
    content, _ = _current_window(tmp_path, "a.py", 1, 2, source_revision(source))
    if clipped:
        content = content.replace("one\ntwo\n", "")
    messages = [{"role": "user", "content": content},
                {"role": "user", "content": "[Source coverage]\na.py (2 lines, unverified): "
                 "all 2 lines are shown below.", "content_sha256": "old"}]
    if clipped:
        with pytest.raises(SourceCoverageClaimError):
            verify_source_coverage_claims(messages, strict=True)
    reconcile_source_coverage(messages)
    coverage = messages[1]["content"]
    assert ("a.py: not shown;" in coverage) == clipped
    assert ("all 2 lines are shown below" in coverage) != clipped
    assert messages[1]["content_sha256"] == hashlib.sha256(coverage.encode()).hexdigest()
    assert verify_source_coverage_claims(messages, strict=True)


def test_off_provider_bytes_match_carried_3b(tmp_path, tmp_path_factory):
    from test_working_set_astra_off_parity import test_off_provider_bytes_match_cutover_base

    test_off_provider_bytes_match_cutover_base(tmp_path, tmp_path_factory, "936a59be20")
