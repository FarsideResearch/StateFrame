"""After an owned-file edit, a pin delivered by a whole read page projects like the fit.

Live P0 (--parallel 2, 0.7.9): a subframe edited its owned source, then read it back.
The read page delivered the post-edit bytes, so the fit converted the ledger pin to a
same_request receipt. The validator's host projection dropped that pin first as
"unbacked" (no Model source window for the path), so the two ledgers differed
("unit-scoped serialized ledger differs from host projection") on every replan.
"""
from __future__ import annotations

import copy
import json
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


# A pin whose only backing is a read page at an OLD (pre-edit) revision.
#
# The fit never asks whether a carrier's revision is current: refresh_read_ranges
# applies drop_delivered_range_pins, which matches path + revision + exact bytes, and
# validate_read_ranges accepts the resulting same_request receipt the same way. So a
# whole, unmodified pre-edit read page still on the wire delivers the pre-edit pin.
# (When the working-set renderer retires that page as stale, it is replaced by a
# "[Source page deferred]" notice; then no carrier remains and both sides drop it.)


def _model_window(revision: str) -> dict[str, object]:
    """A COMPLETE model source window that marks ``revision`` current for PATH."""
    from kernel_request_source_window import MODEL_WINDOW_MARKER, MODEL_WINDOW_SCHEMA

    meta = {"schema": MODEL_WINDOW_SCHEMA, "path": PATH, "revision": revision,
            "absence": "NONE", "status": "COMPLETE"}
    return {"role": "user", "content": MODEL_WINDOW_MARKER + "\n" + json.dumps(meta) + "\n"}


def _old_then_new_pages() -> list[dict[str, object]]:
    messages = _messages(_read_page(PRE_EDIT, PRE_REVISION))
    messages += [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_2", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "owned_source.py"}'}}]},
        {"role": "tool", "tool_call_id": "call_2", "content": _read_page()},
    ]
    return messages


def _assert_old_pin_agrees(messages, frame) -> None:
    fitted = _fit(_ledger(PRE_EDIT, PRE_REVISION), messages, frame)
    row = fitted["reads"][0]
    assert row["range_delivery"] == "same_request"
    assert "pinned_range" not in row
    # The fit's receipt is one the validator's read-range check accepts.
    capacity.validate_read_ranges({"action_ledger": fitted}, messages)

    for with_frame in (True, False):
        projected = _project(_ledger(PRE_EDIT, PRE_REVISION), messages, frame,
                             with_frame=with_frame)
        assert projected == fitted


def test_old_revision_read_page_pin_projects_as_the_fit_converts_it() -> None:
    # Only the pre-edit read page is on the wire; no source window names a revision.
    _assert_old_pin_agrees(_messages(_read_page(PRE_EDIT, PRE_REVISION)), {})


def test_old_revision_read_page_pin_agrees_after_the_post_edit_read() -> None:
    # Pre-edit page, then the post-edit page: the old pin is still exactly delivered.
    _assert_old_pin_agrees(_old_then_new_pages(), {})


def test_old_revision_read_page_pin_agrees_when_a_window_marks_the_new_revision() -> None:
    # The superseded-revision branch: a COMPLETE window makes REVISION current, so the
    # unbacked rule reaches its carrier check for the pre-edit pin.
    messages = [*_old_then_new_pages(), _model_window(REVISION)]
    probe = _ledger(PRE_EDIT, PRE_REVISION)
    capacity.drop_unbacked_range_pins(probe, messages, {})
    assert probe["reads"][0]["pinned_range"] == PRE_EDIT
    _assert_old_pin_agrees(messages, {})


def test_old_revision_pin_is_not_backed_by_a_next_site_at_the_new_revision() -> None:
    # Selected next_site bytes at the current revision cannot back the pre-edit pin.
    frame = {"next_site": {"path": PATH, "revision": REVISION,
                           "lines": {"start": 1, "end": len(LINES)}, "current_bytes": BODY}}
    messages = [*_messages("[read page] unavailable"), _model_window(REVISION)]
    fitted = _fit(_ledger(PRE_EDIT, PRE_REVISION), messages, frame)
    assert "range_delivery" not in fitted["reads"][0]
    projected = _project(_ledger(PRE_EDIT, PRE_REVISION), messages, frame, with_frame=True)
    assert "pinned_range" not in projected["reads"][0]
    assert "range_delivery" not in projected["reads"][0]


def test_stale_deferred_notice_does_not_back_the_old_revision_pin() -> None:
    # The renderer's stale notice names the old revision but carries no bytes.
    notice = ("[Source page deferred]\n"
              f'path="{PATH}" revision={PRE_REVISION} lines=1-{len(LINES)} '
              f"digest={hashlib.sha256(PRE_EDIT.encode()).hexdigest()}\n"
              "These bytes are stale since your edit; re-read current source if needed.")
    messages = [*_messages(_read_page()), {"role": "user", "content": notice}]
    projected = _project(_ledger(PRE_EDIT, PRE_REVISION), messages, {}, with_frame=True)
    assert "pinned_range" not in projected["reads"][0]
    assert "range_delivery" not in projected["reads"][0]


def test_old_revision_page_with_other_bytes_does_not_back_the_pin() -> None:
    # Same old revision label, different body bytes: digest-checked, so not delivered.
    other = PRE_EDIT.replace("return 1", "return 3")
    messages = _messages(_read_page(other, PRE_REVISION))
    fitted = _fit(_ledger(PRE_EDIT, PRE_REVISION), messages, {})
    assert "range_delivery" not in fitted["reads"][0]
    projected = _project(_ledger(PRE_EDIT, PRE_REVISION), messages, {}, with_frame=True)
    assert "pinned_range" not in projected["reads"][0]


def _end_to_end(monkeypatch, tmp_path, *, allocator: str, byte_cap: int = 100_000):
    """Fit a child frame for real, then validate it against the host authority."""
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kernel_frame_allocation_validation as allocation_validation
    import kernel_stateframe_framefit5
    import kernel_unit_policy
    from kernel_frame_optional_allocation import identity
    from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, HARD_MAX_REQUEST_BYTES
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_stateframe_framefit5 import scope_typed_state_for_unit
    from test_parallel_child_scope import _planted_unit

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", allocator)
    path = "pkg/reader.py"
    (tmp_path / "pkg").mkdir()
    (tmp_path / path).write_text(BODY)
    unit = _planted_unit()
    site = {"path": path, "line": 1, "revision": REVISION,
            "lines": {"start": 1, "end": len(LINES)}, "current_bytes": BODY,
            "read_this_request": True}
    ledger = _ledger()
    ledger["reads"][0]["path"] = path
    context_frame = {
        "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
        "task": {"source": "original_task_message", "sha256": "c" * 64},
        "next_site": site, "open_sites": [site], "read_facts": [],
        "applied_edits": [], "last_verdict": {},
        "action_ledger": {"schema": "ledger/1", **ledger, "edits": [], "applied_edits": [],
                          "last_test": None, "superseded_test_count": 0},
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1",
                           "route": "ACTION", "admitted_actions": ["read", "edit"]},
    }
    projected = scope_typed_state_for_unit(copy.deepcopy(context_frame), unit)
    body = {"model": "local", "max_tokens": 100, "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": STATE_HEADER + "\n" + json.dumps(
            {"state": {"framefit5": projected}}, separators=(",", ":"))},
        {"role": "user", "content": "suffix"},
    ]}
    runtime = SimpleNamespace(
        binding=lambda: {"product_revision": "sha256:revision"},
        _request_source_closure={"workspace_authorization": {"scope": "workspace", "generation": "1"}},
        _framefit_action_context={"framefit5": context_frame},
        frame_ceiling=100_000, owner=SimpleNamespace(root=tmp_path),
    )
    if allocator == "on":
        # The allocator's renderer rebuilds delivery truth from whole-line FULL slots.
        from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
        from kernel_working_set_snapshot import freeze_preparation

        prepared = prepare_candidate(freeze_preparation(
            body, SimpleNamespace(owner=SimpleNamespace(root=tmp_path)),
            request_id="next-site-pin", byte_cap=byte_cap))
        assert isinstance(prepared, PreparedCandidate), prepared
        messages = json.loads(prepared.wire)["messages"]
    else:
        # Allocator off: the protected-frame fit, which leaves pins alone without pressure.
        capacity.fit_protected_frame(body, semantic_messages=body["messages"])
        messages = body["messages"]
    capacity.refresh_read_ranges(messages, None)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: SimpleNamespace(spec=unit))
    monkeypatch.setattr(kernel_stateframe_framefit5, "_next_sites", lambda _runtime, _context: [site])
    monkeypatch.setattr("kernel_frame_source_snapshot.frame_source_snapshot",
                        lambda _runtime, _plan: None)
    site_id = identity(site)
    fitted = next(json.loads(row["content"].split("\n", 1)[1])["state"]["framefit5"]
                  for row in messages if str(row.get("content", "")).startswith(STATE_HEADER + "\n"))
    allocation = SimpleNamespace(
        validate=lambda: None,
        identity={"product_revision": "sha256:revision",
                  "workspace_authorization": {"scope": "workspace", "generation": "1"},
                  "ledger_digest": identity(projected["action_ledger"])},
        source_layout={"source_plan": {}},
        capacity={"frame_ceiling": runtime.frame_ceiling,
                  "token_ceiling": ENDPOINT_PROMPT_TOKEN_BUDGET,
                  "host_byte_ceiling": HARD_MAX_REQUEST_BYTES,
                  "template_environment": allocation_validation.template_environment(),
                  "tool_schema_digest": "sha256:tools",
                  "system_messages_digest": identity([messages[0]])},
        site_inventory={"inventory_identity": [site_id], "selected_identities": [site_id],
                        "open_sites": [site_id], "next_site": site_id,
                        "encoding_mode": "full_inventory", "sites": [site]},
        optional_heads={"panel": None, "dependency_revisions": {}},
    )
    allocation_validation.validate_allocation_authority(
        allocation, runtime, messages, check_layout=False)
    return fitted["action_ledger"]["reads"][0]


def test_next_site_only_pin_validates_end_to_end(monkeypatch, tmp_path) -> None:
    """A current-revision pin whose site is next_site: the rendered frame validates.

    The renderer keeps the pinned file as a fixed whole-line slot (or blocks the
    request), so the pin converts on the wire exactly as the projection converts it.
    """
    row = _end_to_end(monkeypatch, tmp_path, allocator="on")
    assert row.get("range_delivery") == "same_request" and "pinned_range" not in row


def test_allocator_off_subframe_ledger_projection_matches_fit(monkeypatch, tmp_path) -> None:
    """Allocator off: an unpressured fit keeps the pin, and so must the projection."""
    row = _end_to_end(monkeypatch, tmp_path, allocator="off")
    assert row.get("pinned_range") == BODY and "range_delivery" not in row
