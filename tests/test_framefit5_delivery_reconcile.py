"""Delivery claims in a headcarry-free frame follow the final frame's source carriers."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import kernel_reread_result_truth
from kernel_stateframe_framefit5 import (
    PRODUCT_MODE,
    STATE_HEADER,
    FrameFit5Error,
    _state_payload,
    validate_product_request,
)

PATH = "django/core/paginator.py"
BODY = "".join(f"line {number}\n" for number in range(1, 6))
REVISION = "sha256:" + hashlib.sha256(BODY.encode("utf-8")).hexdigest()
TASK = "Task: review the paginator."
NOTICE = "[not shown] " + PATH + " re-read to see it"


def _read_page() -> str:
    last = BODY.splitlines(keepends=True)[-1]
    header = (
        f"[read page] path={PATH} revision={REVISION} lines=1-5/5 requested=1-5 "
        f"effective=lines:1-5@0:{len(last)} clip=none chars={len(BODY)} "
        f"bytes={len(BODY.encode('utf-8'))} truncated=false decoded_with_replacement=false"
    )
    return header + "\n\n" + BODY


def _frame(delivery: str, *, headcarry: dict | None = None) -> dict:
    fact = {
        "path": PATH,
        "read_revision": REVISION,
        "current_revision": REVISION,
        "status": "UNCHANGED",
        "unchanged_since_read": True,
        "receipts": [{"status": "PAGE", "revision": REVISION}],
        "body_delivery": delivery,
    }
    if delivery == "not_shown":
        fact["delivery_notice"] = NOTICE
    frame = {
        "mode": PRODUCT_MODE,
        "task": {"sha256": hashlib.sha256(TASK.encode("utf-8")).hexdigest()},
        "action_ledger": {"reads": [], "edits": [], "unclassified_actions": []},
        "read_facts": [fact],
        "applied_edits": [],
        "last_verdict": {"line": "NO GATE RAN", "status": None},
        "open_sites": [],
        "open_work": {"next_site_available": False, "open_edit_site_count": 0},
    }
    if headcarry is not None:
        frame["headcarry"] = headcarry
    return frame


def _messages(frame: dict, tool_result: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a careful code reviewer."},
        {"role": "user", "content": TASK},
        {"role": "user", "content": STATE_HEADER + "\n" + json.dumps({"state": {"framefit5": frame}})},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]\nContinue."},
    ]


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        _active_capture=None,
        current_request_id=None,
        _framefit_action_context={"ledger": {"rows": []}},
    )


def _state_fact(messages: list[dict]) -> dict:
    state_message = next(m for m in messages if str(m["content"]).startswith(STATE_HEADER + "\n"))
    return _state_payload(state_message)[1]["framefit5"]["read_facts"][0]


def test_stale_not_shown_claim_is_reconciled_when_final_tool_page_carries_body():
    messages = _messages(_frame("not_shown"), _read_page())
    validate_product_request(messages, _runtime())
    fact = _state_fact(messages)
    assert fact["body_delivery"] == "present"
    assert "delivery_notice" not in fact


def test_present_claim_without_carrier_is_downgraded_not_dispatched_as_present():
    messages = _messages(_frame("present"), "ok")
    validate_product_request(messages, _runtime())
    fact = _state_fact(messages)
    assert fact["body_delivery"] == "not_shown"
    assert fact["delivery_notice"]


def test_headcarry_path_unchanged():
    messages = _messages(_frame("not_shown", headcarry={}), _read_page())
    validate_product_request(messages, _runtime())
    fact = _state_fact(messages)
    assert fact["body_delivery"] == "present"
    assert "delivery_notice" not in fact


def test_negative_control_without_reconcile_raises_delivery_mismatch(monkeypatch):
    # validate_product_request imports reconcile_frame at call time; the first call is the
    # guard under test. validate_delivery calls it again, so only the first call is disabled.
    real = kernel_reread_result_truth.reconcile_frame
    calls = {"count": 0}

    def skip_first_call(frame, messages):
        calls["count"] += 1
        if calls["count"] == 1:
            return None
        return real(frame, messages)

    monkeypatch.setattr(kernel_reread_result_truth, "reconcile_frame", skip_first_call)
    messages = _messages(_frame("not_shown"), _read_page())
    with pytest.raises(FrameFit5Error, match="read delivery facts do not match"):
        validate_product_request(messages, _runtime())


def test_cache_layout_view_without_headcarry_still_fails_closed(monkeypatch):
    # Cache-layout views are copies, so the reconcile is skipped for them without headcarry.
    # Patched is_cache_layout to True and semantic_messages to a deepcopy (no real cache wire
    # builder is needed for this). Reversal check: with is_cache_layout False the reconcile runs,
    # downgrades the stale "present" claim to not_shown and validation does not raise.
    import copy

    import kernel_cache_layout_view

    monkeypatch.setattr(kernel_cache_layout_view, "is_cache_layout", lambda messages: True)
    monkeypatch.setattr(
        kernel_cache_layout_view, "semantic_messages", lambda messages: copy.deepcopy(messages)
    )
    messages = _messages(_frame("present"), "ok")
    with pytest.raises(FrameFit5Error, match="read delivery facts do not match"):
        validate_product_request(messages, _runtime())


def test_prior_user_context_only_change_reserializes_state_message():
    # reconcile_frame rewrites stale "[not shown] ...: re-read to see it" labels in
    # WORLD KNOWLEDGE bullets to "fresh" when the body is present. read_facts already match
    # the carriers, so only the prior_user_context row changes. Reversal check: comparing only
    # read_facts would see no change, skip the re-serialize, and fail the content assertion.
    import copy

    frame = _frame("present")
    stale_line = f"- {PATH} | [not shown] {PATH}: re-read to see it | read@1\n"
    content = (
        "[kernel] (2 turns) earlier exchanges trimmed to fit context. CAPSULE\n"
        "WORLD KNOWLEDGE (1)\n" + stale_line
    )
    frame["prior_user_context"] = [
        {"role": "user", "content": content, "content_sha256": "stale"},
    ]
    messages = _messages(frame, _read_page())
    facts_before = copy.deepcopy(_state_fact(messages))
    state_before = messages[2]["content"]

    validate_product_request(messages, _runtime())

    assert _state_fact(messages) == facts_before
    assert messages[2]["content"] != state_before
    state = _state_payload(messages[2])[1]["framefit5"]
    row = state["prior_user_context"][0]
    assert f"{PATH} | fresh | read@1" in row["content"]
    assert "[not shown]" not in row["content"]
    assert row["content_sha256"] == hashlib.sha256(row["content"].encode("utf-8")).hexdigest()


def test_allocator_on_without_certificate_reconciles_stale_claim(monkeypatch):
    # Allocator on but no working-set certificate: the bytes are not certified, so the
    # reconcile must still run and correct the stale "not_shown" claim.
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    runtime = _runtime()
    assert runtime._active_capture is None
    messages = _messages(_frame("not_shown", headcarry={}), _read_page())
    state_before = messages[2]["content"]
    validate_product_request(messages, runtime)
    fact = _state_fact(messages)
    assert fact["body_delivery"] == "present"
    assert "delivery_notice" not in fact
    assert messages[2]["content"] != state_before


def test_allocator_on_with_certificate_does_not_rewrite_state_message(monkeypatch):
    # Certified path: verify_messages is the proof of record, so the StateFrame message must
    # pass through byte-identical even though its claim would otherwise be reconciled.
    import kernel_frame_allocation_validation
    import kernel_working_set_proof

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    certificate = object()
    verified: list[tuple[object, object]] = []
    monkeypatch.setattr(
        kernel_working_set_proof,
        "verify_messages",
        lambda msgs, cert: verified.append((msgs, cert)),
    )
    monkeypatch.setattr(
        kernel_frame_allocation_validation, "require_active_allocation", lambda *a, **k: None
    )
    monkeypatch.setattr(
        kernel_frame_allocation_validation, "validate_allocation_authority", lambda *a, **k: None
    )
    runtime = _runtime()
    runtime._active_capture = SimpleNamespace(_working_set_certificate=certificate)
    messages = _messages(_frame("not_shown"), _read_page())
    state_before = messages[2]["content"]
    validate_product_request(messages, runtime, frame_selection=object())
    assert len(verified) == 1
    assert verified[0][1] is certificate
    assert messages[2]["content"] == state_before
