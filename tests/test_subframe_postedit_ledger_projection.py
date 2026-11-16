"""After an owned-file edit, a pin delivered by a whole read page projects like the fit.

Live P0 (--parallel 2, 0.7.9): a subframe edited its owned source, then read it back.
The read page delivered the post-edit bytes, so the fit converted the ledger pin to a
same_request receipt. The validator's host projection dropped that pin first as
"unbacked" (no Model source window for the path), so the two ledgers differed
("unit-scoped serialized ledger differs from host projection") on every replan.
"""
from __future__ import annotations

import copy
import hashlib

import kernel_stateframe_capacity as capacity

PATH = "owned_source.py"
BODY = '"""Helpers."""\n\n\ndef value():\n    return 2\n'
LINES = BODY.splitlines(keepends=True)
REVISION = "sha256:" + hashlib.sha256(BODY.encode("utf-8")).hexdigest()
PRE_EDIT = '"""Helpers."""\n\n\ndef value():\n    return 1\n'
PRE_REVISION = "sha256:" + hashlib.sha256(PRE_EDIT.encode("utf-8")).hexdigest()


def _read_page(body: str = BODY, revision: str = REVISION) -> str:
    lines = body.splitlines(keepends=True)
    header = (
        f"[read page] path={PATH} revision={revision} lines=1-{len(lines)}/{len(lines)} "
        f"requested=lines:1-120 effective=lines:1-{len(lines)}@0:{len(lines[-1])} clip=none "
        f"chars={len(body)} bytes={len(body.encode('utf-8'))} truncated=false "
        "decoded_with_replacement=false"
    )
    return header + "\n\n" + body


def _messages(page: str) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Task: fix the helper."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "owned_source.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": page},
    ]


def _ledger(pin: str = BODY, revision: str = REVISION) -> dict[str, object]:
    return {"reads": [{
        "kind": "read", "path": PATH, "revision": revision, "call_ids": ["call_1"],
        "lines": {"start": 1, "end": len(pin.splitlines()), "total": len(pin.splitlines())},
        "content_sha256": hashlib.sha256(pin.encode("utf-8")).hexdigest(),
        "pinned_range": pin,
    }]}


def _fit(ledger: dict[str, object], messages, frame) -> dict[str, object]:
    """The fit's delivery rule (refresh_read_ranges with the allocator on)."""
    fitted = copy.deepcopy(ledger)
    capacity.drop_delivered_range_pins(fitted, messages, frame)
    return fitted


def _project(ledger: dict[str, object], messages, frame, *, with_frame: bool) -> dict[str, object]:
    """The allocation validator's host projection of the same ledger."""
    projected = copy.deepcopy(ledger)
    if with_frame:
        capacity.drop_unbacked_range_pins(projected, messages, frame)
    else:
        capacity.drop_unbacked_range_pins(projected, messages)
    capacity.drop_delivered_range_pins(projected, messages, frame)
    return projected


def test_post_edit_read_page_pin_projects_as_the_fit_converts_it() -> None:
    """RED on 0.7.9: the projection dropped the pin without a same_request receipt."""
    messages = _messages(_read_page())
    fitted = _fit(_ledger(), messages, {})
    assert fitted["reads"][0]["range_delivery"] == "same_request"
    assert "pinned_range" not in fitted["reads"][0]

    assert _project(_ledger(), messages, {}, with_frame=False) == fitted


def test_next_site_delivered_pin_projects_as_the_fit_converts_it() -> None:
    frame = {"next_site": {"path": PATH, "revision": REVISION,
                           "lines": {"start": 1, "end": len(LINES)}, "current_bytes": BODY}}
    messages = _messages("[read page] unavailable")
    fitted = _fit(_ledger(), messages, frame)
    assert fitted["reads"][0]["range_delivery"] == "same_request"

    assert _project(_ledger(), messages, frame, with_frame=True) == fitted


def test_superseded_pin_is_still_dropped_when_only_the_new_revision_is_delivered() -> None:
    # The pre-edit pin's bytes are not on the wire: the unbacked rule still drops it.
    messages = _messages(_read_page())
    projected = _project(_ledger(PRE_EDIT, PRE_REVISION), messages, {}, with_frame=True)
    row = projected["reads"][0]
    assert "pinned_range" not in row
    assert "range_delivery" not in row


def test_pin_with_mismatched_digest_is_not_backed_by_a_read_page() -> None:
    messages = _messages(_read_page())
    ledger = _ledger()
    ledger["reads"][0]["content_sha256"] = "0" * 64
    capacity.drop_unbacked_range_pins(ledger, messages)
    assert "pinned_range" not in ledger["reads"][0]
