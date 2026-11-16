"""Exact range pins are dropped only when their bytes are delivered in the same request."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import json

from kernel_request_source_window import MODEL_WINDOW_MARKER, MODEL_WINDOW_SCHEMA, WINDOW_CONTENT_MARKER, WINDOW_END_MARKER
from test_parallel_site_refs import _site, _window

import kernel_stateframe_capacity as capacity


def _complete_window(site: dict[str, object], source: str) -> dict[str, object]:
    """A whole-file window carrying the COMPLETE status the producer's backing rule reads."""
    text = source
    metadata = {
        "schema": MODEL_WINDOW_SCHEMA, "path": site["path"], "revision": site["revision"],
        "absence": "NONE", "status": "COMPLETE",
        "ranges": [{"start_line": 1, "end_line": len(source.splitlines()),
                    "trailing_newline": text.endswith("\n")}],
    }
    body = (f"[Current source: {site['path']}]\n{MODEL_WINDOW_MARKER}\n"
            + json.dumps(metadata, separators=(",", ":")) + f"\n{WINDOW_CONTENT_MARKER}\n"
            + text + WINDOW_END_MARKER)
    return {"role": "user", "content": body}


def _pinned_row(site: dict[str, object], pin: str, start: int, end: int) -> dict[str, object]:
    return {
        "kind": "read", "path": site["path"], "revision": site["revision"],
        "lines": {"start": start, "end": end, "total": site["lines"]["total"]},
        "content_sha256": hashlib.sha256(pin.encode()).hexdigest(),
        "pinned_range": pin,
    }


def test_delivered_pin_is_dropped_and_undelivered_pin_is_kept(tmp_path: Path) -> None:
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    other = _site("pkg/other.py", source)
    ledger = {"reads": [
        _pinned_row(site, "alpha\nbeta\ngamma\n", 1, 3),
        _pinned_row(other, "alpha\nbeta\ngamma\n", 1, 3),
    ]}
    messages = [{"role": "user", "content": _window(site, source, 1, 3)["content"]}]

    capacity.drop_delivered_range_pins(ledger, messages, {})

    delivered, undelivered = ledger["reads"]
    assert "pinned_range" not in delivered
    assert delivered["range_delivery"] == "same_request"
    assert undelivered["pinned_range"] == "alpha\nbeta\ngamma\n"
    assert "range_delivery" not in undelivered


def test_pin_with_digest_mismatch_is_kept_even_when_delivered() -> None:
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    row = _pinned_row(site, "alpha\nbeta\ngamma\n", 1, 3)
    row["content_sha256"] = "0" * 64
    ledger = {"reads": [row]}
    messages = [{"role": "user", "content": _window(site, source, 1, 3)["content"]}]

    capacity.drop_delivered_range_pins(ledger, messages, {})

    assert ledger["reads"][0]["pinned_range"] == "alpha\nbeta\ngamma\n"


def test_pin_for_superseded_revision_is_dropped_even_without_delivery() -> None:
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    row = _pinned_row(site, "alpha\nbeta\ngamma\n", 1, 3)
    ledger = {"reads": [row]}
    edited = "alpha\nchanged\ngamma\n"
    current_site = _site("pkg/reader.py", edited)
    messages = [_complete_window(current_site, edited)]

    capacity.drop_unbacked_range_pins(ledger, messages)

    assert "pinned_range" not in ledger["reads"][0]


def test_pin_for_current_revision_is_kept_by_the_unbacked_rule() -> None:
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    ledger = {"reads": [_pinned_row(site, "alpha\nbeta\ngamma\n", 1, 3)]}
    messages = [_complete_window(site, source)]

    capacity.drop_unbacked_range_pins(ledger, messages)

    assert ledger["reads"][0]["pinned_range"] == "alpha\nbeta\ngamma\n"


def _frame_with(row: dict[str, object]) -> dict[str, object]:
    """Minimal frame-only state carrying one read receipt (no same_request row yet)."""
    from kernel_stateframe_framefit5 import PRODUCT_MODE

    return {"mode": PRODUCT_MODE, "action_ledger": {
        "schema": "stateframe-action-ledger/1", "reads": [row], "edits": [],
        "unclassified_actions": [], "superseded_test_count": 0, "applied_edits": [],
        "last_test": None,
    }}


def _state_message(frame: dict[str, object]) -> dict[str, object]:
    from kernel_stateframe_framefit5 import _serialize_state_message

    return {"role": "user", "content": _serialize_state_message({"state": {"framefit5": frame}})}


def _frame_of(message: dict[str, object]) -> dict[str, object]:
    return json.loads(str(message["content"]).split("\n", 1)[1])["state"]["framefit5"]


def test_allocator_on_refresh_converts_delivered_pin_without_prior_receipt(monkeypatch) -> None:
    """RED (fu070 red 2): a delivered current-revision pin with no same_request row must convert.

    Otherwise the wire frame keeps pinned_range while the validator projection converts it,
    and every replacement request fails the ledger comparison.
    """
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    row = _pinned_row(site, source, 1, 3)
    message = _state_message(_frame_with(row))
    messages = [{"role": "user", "content": _window(site, source, 1, 3)["content"]}, message]

    expected = copy.deepcopy(_frame_of(message)["action_ledger"])
    capacity.drop_delivered_range_pins(expected, messages, {})
    capacity.refresh_read_ranges(messages, {})
    actual = _frame_of(message)["action_ledger"]

    assert actual == expected
    assert "pinned_range" not in actual["reads"][0]
    assert actual["reads"][0]["range_delivery"] == "same_request"


def test_allocator_off_refresh_leaves_frame_byte_identical(monkeypatch) -> None:
    """Allocator OFF keeps the pre-FU-070 wire: the same delivered pin is not rewritten."""
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    message = _state_message(_frame_with(_pinned_row(site, source, 1, 3)))
    before = str(message["content"])
    messages = [{"role": "user", "content": _window(site, source, 1, 3)["content"]}, message]

    capacity.refresh_read_ranges(messages, {})

    assert message["content"] == before
