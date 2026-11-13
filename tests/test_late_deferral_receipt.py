"""A deferral discovered after the first receipt is not a page lost by owner admission.

Regression (endurance s7_m1, v0.5.9): window fitting deferred the ticket target after
the frame receipt was already satisfied.  The post-owner re-verify compared the new
(unsatisfied, deferral-carrying) receipt against that stale satisfied receipt and raised
``final source delivery lost after owner admission`` although the admitted request
kept every page and the ``read`` tool.  The engine stopped with ENGINE_ERROR while
preparing request 14.

The reconciliation must raise only when the admitted request really dropped a verified
page or the read route; a late deferral is journaled as a typed harden8 advisory.
"""
from types import SimpleNamespace

import pytest

import kernel_request_source_window as window

TARGET = "core/__init__.py"
SIBLING = "core/service.py"
REVISION = "sha256:" + "0" * 64
READ_TOOL = {"type": "function", "function": {
    "name": "read", "parameters": {"type": "object", "properties": {}}}}
EDIT_TOOL = {"type": "function", "function": {
    "name": "edit", "parameters": {"type": "object", "properties": {}}}}


def _runtime():
    events = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    return SimpleNamespace(journal=journal), events


def _complete_receipt(*paths):
    return {
        "status": "COMPLETE", "proof_status": "VERIFIED", "proof_failures": [],
        "deferred_paths": [], "deferred_notice_verified": False,
        "read_tool_available": True,
        "paths": {path: {"status": "COMPLETE", "proof_status": "VERIFIED", "revision": REVISION}
                  for path in paths},
    }


def _late_deferral_receipt(*, deferred, kept=(), read_tool_available=True, units=0):
    rows = {path: {"status": "COMPLETE", "proof_status": "VERIFIED", "revision": REVISION}
            for path in kept}
    for path in deferred:
        rows[path] = {
            "status": "DEFERRED", "proof_status": "VERIFIED", "revision": REVISION,
            "retrieval_route": {
                "status": "EXECUTABLE_BOUNDED_READ" if units else "NO_UNITS",
                "tool": "read", "path": path, "revision": REVISION,
                "unit_handles": [f"unit:{path}:{index}" for index in range(units)],
            },
        }
    return {
        "status": "DEFERRED", "proof_status": "VERIFIED", "proof_failures": [],
        "deferred_paths": list(deferred), "deferred_notice_verified": True,
        "read_tool_available": read_tool_available, "paths": rows,
    }


def test_late_deferral_refreshes_receipt():
    # Same pages as before (the target moved into the deferral set) and read admitted:
    # the regression is a late deferral, journaled, never a crash.
    runtime, events = _runtime()
    capture = SimpleNamespace()
    previous = _complete_receipt(TARGET, SIBLING)
    final = _late_deferral_receipt(deferred=(TARGET,), kept=(SIBLING,), units=0)
    assert not window.source_delivery_satisfied(final)
    assert window.source_delivery_satisfied(previous)

    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, previous, final, tools=[READ_TOOL, EDIT_TOOL])

    assert verdict["genuine_loss"] is False
    assert verdict["reason"] == "late_deferral_unsatisfied"
    assert verdict["lost_paths"] == []
    assert verdict["late_deferred_paths"] == [TARGET]
    assert verdict["read_tool_admitted"] is True
    assert verdict["deferred_rows"][0]["unit_count"] == 0
    assert capture.request_source_delivery_regression == verdict
    kinds = [kind for kind, _ in events]
    assert kinds == ["harden8_advisory_failure"]
    fields = events[0][1]
    assert fields["boundary"] == "completion_owner.source_delivery_receipt"
    assert fields["failure_class"] == "ADVISORY"
    assert fields["disposition"] == "degrade"
    assert "late_deferral_unsatisfied" in fields["message"]
    assert TARGET in fields["message"]


def test_late_deferral_real_loss_still_raises():
    # A page COMPLETE before admission that is neither delivered nor deferred afterwards
    # is a genuine loss: keep raising, and name the page.
    runtime, events = _runtime()
    capture = SimpleNamespace()
    previous = _complete_receipt(TARGET, SIBLING)
    final = _late_deferral_receipt(deferred=(TARGET,), kept=(), units=0)
    with pytest.raises(RuntimeError, match="lost after owner admission") as info:
        window.reconcile_owner_source_delivery(
            runtime, capture, previous, final, tools=[READ_TOOL])
    assert SIBLING in str(info.value)
    assert capture.request_source_delivery_regression["lost_paths"] == [SIBLING]
    assert events == []


def test_read_tool_dropped_by_admission_still_raises():
    # Deferred pages need the read route; an admission that dropped `read` lost it.
    runtime, _events = _runtime()
    capture = SimpleNamespace()
    previous = _complete_receipt(TARGET, SIBLING)
    final = _late_deferral_receipt(
        deferred=(TARGET,), kept=(SIBLING,), read_tool_available=False, units=3)
    with pytest.raises(RuntimeError, match="read tool dropped by admission"):
        window.reconcile_owner_source_delivery(
            runtime, capture, previous, final, tools=[EDIT_TOOL])
    assert capture.request_source_delivery_regression["read_tool_dropped"] is True


def test_satisfied_or_never_satisfied_receipts_are_not_reconciled():
    runtime, events = _runtime()
    capture = SimpleNamespace()
    assert window.reconcile_owner_source_delivery(
        runtime, capture, _complete_receipt(TARGET), _complete_receipt(TARGET)) is None
    unsatisfied_before = _late_deferral_receipt(deferred=(TARGET,), units=0)
    assert window.reconcile_owner_source_delivery(
        runtime, capture, unsatisfied_before,
        _late_deferral_receipt(deferred=(TARGET,), units=0)) is None
    assert window.reconcile_owner_source_delivery(
        runtime, capture, None, _late_deferral_receipt(deferred=(TARGET,), units=0)) is None
    assert events == []
    assert not hasattr(capture, "request_source_delivery_regression")


def test_envelope_over_budget_target_page_defers_without_crash(tmp_path):
    # Non-empty sibling of the zero-byte case: the envelope already exceeds the window
    # budget, so window fitting defers the target after the frame receipt was computed.
    # The real wire receipt carries the deferral as a typed proof failure; against a
    # stale satisfied receipt the reconciliation must journal, not raise. Only a page
    # that is neither delivered nor deferred anywhere is a loss.
    root = tmp_path / "repo"
    (root / "core").mkdir(parents=True)
    target = root / "core/service.py"
    target.write_text(
        "".join(f"def handler_{index}(value):\n    return value + {index}\n\n"
                for index in range(60)),
        encoding="utf-8",
    )
    runtime = SimpleNamespace(
        target=target, owner=SimpleNamespace(root=root), product_revision="0")
    runtime.journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    events = []
    messages = [{"role": "user", "content": "x" * 4000}]
    tools = [READ_TOOL, EDIT_TOOL]
    result = window.bound_runtime_sources(
        messages, runtime, max_request_bytes=512, tools=tools)
    assert isinstance(result, dict), result
    plan = runtime._request_source_closure
    assert plan["deferred_paths"] == [SIBLING]
    assert not any(
        window.parse_model_source_window(message.get("content")) is not None
        for message in messages if isinstance(message, dict)
    )
    window.upsert_deferred_source_notice(messages, plan["deferred_paths"])
    final = window.verify_final_source_delivery(
        plan, messages, payload={"messages": messages, "tools": tools},
        sources={SIBLING: target.read_text(encoding="utf-8")},
    )
    assert final["deferred_notice_verified"] is True
    assert final["read_tool_available"] is True
    assert not window.source_delivery_satisfied(final)
    assert any(failure["kind"] == "deferred_source_path_unselected"
               for failure in final["proof_failures"]), final["proof_failures"]

    capture = SimpleNamespace()
    verdict = window.reconcile_owner_source_delivery(
        runtime, capture, _complete_receipt(SIBLING), final, tools=tools)
    assert verdict["genuine_loss"] is False
    assert verdict["reason"] == "deferral_unqualifiable"
    assert verdict["unqualified_deferred_paths"] == [SIBLING]
    assert [kind for kind, _ in events if kind.startswith("harden8")] == ["harden8_advisory_failure"]

    # Control: a second page that was COMPLETE before and is now nowhere is a loss.
    with pytest.raises(RuntimeError, match=TARGET):
        window.reconcile_owner_source_delivery(
            runtime, capture, _complete_receipt(SIBLING, TARGET), final, tools=tools)
