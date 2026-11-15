"""Child edit-site references require exact source bytes on the same wire."""
from __future__ import annotations

import hashlib
import json
import math
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_stateframe_framefit5 as framefit5
from kernel_request_source_window import MODEL_WINDOW_MARKER, MODEL_WINDOW_SCHEMA, WINDOW_CONTENT_MARKER, WINDOW_END_MARKER
from kernel_stateframe_envelope import serialize_stateframe_payload
from kernel_completion_owner import _wire_continuation_limits
from kernel_frame_optional_allocation import identity
from kernel_parallel_records import default_subframe_token_cap
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, HARD_MAX_REQUEST_BYTES
from kernel_frame_allocation_validation import validate_allocation_authority, template_environment
import kernel_unit_policy


def _site(path: str, source: str, start: int = 1, end: int | None = None) -> dict[str, object]:
    lines = source.splitlines(keepends=True)
    end = len(lines) if end is None else end
    excerpt = "".join(lines[start - 1:end])
    return {
        "path": path, "line": start, "lines": {"start": start, "end": end, "total": len(lines)},
        "revision": "sha256:" + hashlib.sha256(source.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "current_bytes": excerpt, "current_line": lines[start - 1].rstrip("\r\n"),
        "source": "remaining_sites_inventory",
    }


def _window(site: dict[str, object], source: str, start: int, end: int) -> dict[str, object]:
    text = "".join(source.splitlines(keepends=True)[start - 1:end])
    metadata = {
        "schema": MODEL_WINDOW_SCHEMA, "path": site["path"], "revision": site["revision"],
        "ranges": [{"start_line": start, "end_line": end,
                    "trailing_newline": text.endswith("\n")}],
    }
    body = (f"[Current source: {site['path']}]\n{MODEL_WINDOW_MARKER}\n"
            + json.dumps(metadata, separators=(",", ":")) + f"\n{WINDOW_CONTENT_MARKER}\n"
            + f"--- source lines {start}-{end} ---\n" + text
            + ("" if text.endswith("\n") else "\n") + WINDOW_END_MARKER)
    return {"role": "user", "content": body}


def test_site_ref_or_inline_requires_full_matching_same_request_window():
    source = "alpha\nbeta\ngamma\n"
    site = _site("pkg/reader.py", source)
    full = _window(site, source, 1, 3)
    referenced = framefit5.site_ref_or_inline(site, [full])
    assert "current_bytes" not in referenced
    assert referenced["source_ref"] == {
        "path": site["path"], "lines": {"start": 1, "end": 3},
        "content_sha256": site["content_sha256"],
    }
    assert framefit5.site_ref_or_inline(site, [_window(site, source, 1, 2)]) == site
    assert framefit5.site_ref_or_inline(site, []) == site
    mismatched = _window(site, "alpha\nWRONG\ngamma\n", 1, 3)
    assert framefit5.site_ref_or_inline(site, [mismatched]) == site
    bad_digest = {**site, "content_sha256": "0" * 64}
    assert framefit5.site_ref_or_inline(bad_digest, [full]) == bad_digest


def test_reference_validation_reads_current_workspace_bytes(tmp_path: Path):
    path = tmp_path / "pkg" / "reader.py"
    path.parent.mkdir()
    path.write_text("alpha\nbeta\ngamma\n")
    source = path.read_text()
    site = _site("pkg/reader.py", source)
    referenced = framefit5.site_ref_or_inline(site, [_window(site, source, 1, 3)])
    framefit5._verify_next_site_bytes(referenced, tmp_path)
    path.write_text("alpha\nchanged\ngamma\n")
    with pytest.raises(framefit5.FrameFit5Error):
        framefit5._verify_next_site_bytes(referenced, tmp_path)


def test_final_wire_fit_inlines_an_unmatched_reference_and_rejects_stale_bytes(
        tmp_path: Path) -> None:
    path = tmp_path / "pkg" / "reader.py"
    path.parent.mkdir()
    source = "alpha\nbeta\ngamma\n"
    path.write_text(source)
    site = _site("pkg/reader.py", source)
    reference = framefit5.site_ref_or_inline(site, [_window(site, source, 1, 3)])
    assert "source_ref" in reference
    fitted_window = _window(site, source, 1, 2)
    original_messages = [
        {"role": "user", "content": serialize_stateframe_payload({
            "schema": "stateframe-f4/1", "state": {"framefit5": {
                "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
                "subframe_scope": {"schema": "stateframe-subframe-state-scope/1"},
                "next_site": reference, "open_sites": [reference],
            }},
        })},
        fitted_window,
    ]
    stale_messages = copy.deepcopy(original_messages)

    changed = framefit5._inline_unmatched_site_references(original_messages, tmp_path)

    assert changed is True
    frame = json.loads(original_messages[0]["content"].split("\n", 1)[1])["state"]["framefit5"]
    inlined = frame["open_sites"][0]
    assert "source_ref" not in inlined
    assert inlined["current_bytes"] == source
    assert frame["next_site"] == inlined
    framefit5._verify_next_site_bytes(inlined, tmp_path)

    path.write_text("alpha\nchanged\ngamma\n")
    with pytest.raises(framefit5.FrameFit5Error):
        framefit5._inline_unmatched_site_references(stale_messages, tmp_path)


def test_child_owner_protocol_carries_live_continuation_limits(monkeypatch):
    line = "[CONTINUATION LIMITS | stateframe-continuation-limits/1; decisions=4]"
    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
             "owner_protocol": {"schema": "stateframe-framefit5-owner/1"},
             "subframe_scope": {"schema": "stateframe-subframe-state-scope/1"}}
    payload = {"schema": "stateframe-f4/1", "state": {"framefit5": frame}}
    message = {"role": "user", "content": serialize_stateframe_payload(payload)}
    runtime = SimpleNamespace(
        continuation_limits=SimpleNamespace(render_continuation_limits=lambda: line),
        _framefit_action_context={"framefit5": dict(frame)},
    )
    monkeypatch.setattr(framefit5, "_active_unit_spec", lambda: object())
    limit_message = {"role": "user", "content": line}
    result = framefit5.attach_owner_protocol(
        [message, limit_message], [message, limit_message], runtime, {})
    projected = json.loads(result[0]["content"].split("\n", 1)[1])["state"]["framefit5"]
    assert projected["owner_protocol"]["continuation_limits"] == line
    assert runtime._framefit_action_context["framefit5"]["owner_protocol"]["continuation_limits"] == line

    unit = SimpleNamespace(to_json=lambda: {
        "unit_id": "a", "contract": {
            "id": "a", "owns": [], "creates": [], "reads": [], "requirement_ids": [],
            "check": {"command_kind": "none", "selectors": [], "source_paths": []},
        }, "requirement_clauses": [],
    })
    scoped = framefit5.scope_typed_state_for_unit(
        {**frame, "open_sites": [], "next_site": None}, unit,
        delivered_windows=[{"role": "user", "content": line}],
        continuation_limits="[CONTINUATION LIMITS | stale]",
    )
    assert scoped["owner_protocol"]["continuation_limits"] == line


def test_final_wire_carries_child_limit_without_changing_parent_bytes():
    line = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] decisions=4"
    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
             "owner_protocol": {"schema": "stateframe-framefit5-owner/1"}}
    def messages(child: bool) -> list[dict[str, object]]:
        local = {**frame}
        if child:
            local["subframe_scope"] = {"schema": "stateframe-subframe-state-scope/1"}
        return [
            {"role": "user", "content": serialize_stateframe_payload(
                {"schema": "stateframe-f4/1", "state": {"framefit5": local}})},
            {"role": "user", "content": line},
        ]
    parent = messages(False)
    original = copy.deepcopy(parent)
    assert framefit5.carry_scoped_continuation_limits(parent) is False
    assert parent == original
    child = messages(True)
    assert framefit5.carry_scoped_continuation_limits(child) is True
    result = json.loads(child[0]["content"].split("\n", 1)[1])
    assert result["state"]["framefit5"]["owner_protocol"]["continuation_limits"] == line
    assert framefit5.carry_scoped_continuation_limits(child) is False


def test_scoped_budget_reader_uses_serialized_limit_when_cache_frame_is_partial():
    line = "[CONTINUATION LIMITS | stateframe-continuation-limits/1] decisions=4"
    frame = {"schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
             "owner_protocol": {"schema": "stateframe-framefit5-owner/1"},
             "subframe_scope": {"schema": "stateframe-subframe-state-scope/1"}}
    def wire(value: dict[str, object]) -> list[dict[str, object]]:
        return [{"role": "user", "content": serialize_stateframe_payload(
            {"schema": "stateframe-f4/1", "state": {"framefit5": value}})},
            {"role": "user", "content": line}]
    assert _wire_continuation_limits(wire(frame)) == line
    assert _wire_continuation_limits(wire({**frame, "subframe_scope": None})) is None


def test_allocation_compares_reference_identity_from_full_host_site(monkeypatch, tmp_path: Path):
    path = tmp_path / "pkg" / "reader.py"
    path.parent.mkdir()
    source = "".join(f"value_{index} = {index}\n" for index in range(1, 82))
    path.write_text(source)
    site = _site("pkg/reader.py", source, 1, 80)
    frame = {
        "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
        "task": {"source": "original_task_message", "sha256": "a" * 64},
        "open_sites": [site], "next_site": site, "action_ledger": {},
        "last_verdict": {}, "applied_edits": [], "read_facts": [],
    }
    unit = SimpleNamespace(to_json=lambda: {
        "unit_id": "reader", "contract": {
            "id": "reader", "owns": ["pkg/reader.py"], "creates": [], "reads": [],
            "requirement_ids": [],
            "check": {"command_kind": "none", "selectors": [], "source_paths": []},
        }, "requirement_clauses": [],
    })
    window = _window(site, source, 1, 81)
    wire_frame = framefit5.scope_typed_state_for_unit(
        frame, unit, root=tmp_path, delivered_windows=[window])
    wire_site = wire_frame["open_sites"][0]
    assert "source_ref" in wire_site and identity(wire_site) != identity(site)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": serialize_stateframe_payload(
            {"schema": "stateframe-f4/1", "state": {"framefit5": wire_frame}})},
        window,
    ]
    authorization = {"scope": "workspace", "generation": "1"}
    runtime = SimpleNamespace(
        binding=lambda: {"product_revision": "sha256:revision"},
        _request_source_closure={"workspace_authorization": authorization},
        _framefit_action_context={"framefit5": frame}, frame_ceiling=100_000,
        owner=SimpleNamespace(root=tmp_path),
    )
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: SimpleNamespace(spec=unit))
    monkeypatch.setattr(framefit5, "_next_sites", lambda _runtime, _context: [site])
    monkeypatch.setattr(framefit5, "_journal_read_facts", lambda _context, _root, _ledger: [])
    monkeypatch.setattr("kernel_frame_source_snapshot.frame_source_snapshot",
                        lambda _runtime, _plan: None)
    key = identity(wire_site)
    allocation = SimpleNamespace(
        validate=lambda: None,
        identity={"product_revision": "sha256:revision",
                  "workspace_authorization": authorization,
                  "ledger_digest": identity(wire_frame["action_ledger"])},
        source_layout={"source_plan": {}},
        capacity={"frame_ceiling": runtime.frame_ceiling,
                  "token_ceiling": ENDPOINT_PROMPT_TOKEN_BUDGET,
                  "host_byte_ceiling": HARD_MAX_REQUEST_BYTES,
                  "template_environment": template_environment(),
                  "tool_schema_digest": "sha256:tools",
                  "system_messages_digest": identity([messages[0]])},
        site_inventory={"inventory_identity": [key], "selected_identities": [key],
                        "open_sites": [key], "next_site": key,
                        "encoding_mode": "full_inventory", "sites": [wire_site]},
        optional_heads={"panel": None, "dependency_revisions": {}},
    )
    validate_allocation_authority(allocation, runtime, messages, check_layout=False)


@pytest.mark.parametrize("adapter,count", [
    ("csv", 0), ("fixed", 7), ("json", 8), ("ini", 15),
])
def test_adapter_sized_first_child_requests_fit_default_estimate(adapter: str, count: int):
    """Measure complete synthetic wires with the adapter campaign's site cardinalities."""
    paths = [f"src/adapters4/adapters/{adapter}_{part}.py" for part in ("reader", "writer")]
    sources = {
        path: "".join(f"line_{number:03d} = value_{number:03d}  # " + "x" * 22 + "\n"
                      for number in range(1, 101))
        for path in paths
    }
    sites = [
        _site(paths[index % 2], sources[paths[index % 2]],
              start=1 + index % 12, end=80 + index % 12)
        for index in range(count)
    ]
    windows = [_window(_site(path, source), source, 1, 100)
               for path, source in sources.items()]
    unit = SimpleNamespace(to_json=lambda: {
        "unit_id": f"{adapter}_adapter",
        "contract": {"id": f"{adapter}_adapter", "outcome": "Parse fixture data",
                     "owns": paths, "creates": [], "reads": [], "requirement_ids": [],
                     "check": {"command_kind": "pytest", "selectors": [], "source_paths": []}},
        "requirement_clauses": [],
    })
    state = {
        "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
        "task": {"source": "original_task_message", "sha256": "a" * 64},
        "open_sites": sites, "next_site": sites[0] if sites else None,
        "action_ledger": {}, "last_verdict": {},
    }
    def wire(frame: dict[str, object]) -> dict[str, object]:
        payload = {"schema": "stateframe-f4/1", "state": {"framefit5": frame}}
        return {
            "messages": [
                {"role": "system", "content": "S" * 4800},
                {"role": "user", "content": "T" * 3300},
                {"role": "user", "content": serialize_stateframe_payload(payload)},
                *windows,
                {"role": "user", "content": "V" * 3500},
            ],
            "tools": [{"type": "function", "function": {
                "name": "read", "description": "D" * 8800,
                "parameters": {"type": "object", "properties": {}},
            }}],
        }
    before = wire(state)
    scoped = framefit5.scope_typed_state_for_unit(state, unit, delivered_windows=before["messages"])
    after = wire(scoped)
    def estimated_tokens(body: dict[str, object]) -> int:
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        return math.ceil(len(encoded) * 0.3193)
    before_tokens, after_tokens = estimated_tokens(before), estimated_tokens(after)
    print(f"ADAPTER_FIRST_REQUEST {adapter}: before={before_tokens} after={after_tokens} "
          f"refs={sum('source_ref' in site for site in scoped['open_sites'])}/{count}")
    assert all("source_ref" in site for site in scoped["open_sites"])
    assert after_tokens <= default_subframe_token_cap(ENDPOINT_PROMPT_TOKEN_BUDGET)
    if count:
        assert after_tokens < before_tokens
