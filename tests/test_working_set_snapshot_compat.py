"""Recorded overlays and closure windows retain exact immutable source bytes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_working_set_snapshot import freeze_preparation
from kernel_working_set_snapshot_compat import snapshot_semantic_messages


def test_recorded_overlay_reconstructs_source_references_without_mutating_wire():
    path = Path.home() / "crew-lanes/tmp/crew-01/ret/r38g/run/wire/0020.request.json"
    if not path.is_file():
        pytest.skip("recorded r38g wire absent")
    body = json.loads(path.read_bytes())
    before = copy.deepcopy(body)
    messages = snapshot_semantic_messages(body["messages"])
    frame = json.loads(next(m["content"] for m in messages
        if str(m.get("content", "")).startswith("[StateFrame]\n")).split("\n", 1)[1])
    site = frame["state"]["framefit5"]["next_site"]
    page = next(m["content"] for m in messages if m.get("role") == "tool"
                and str(m.get("content", "")).startswith("[read page]"))
    payload = page.split("\n", 2)[2]
    assert site["current_bytes"] == payload
    assert body == before


def test_closure_slots_exclude_grammar_and_keep_disjoint_ranges():
    metadata = {"schema": "stateframe-request-source-window/1-projection/1",
        "path": "pkg/a.py", "revision": "sha256:" + "1" * 64, "owner": "closure",
        "status": "CONTINUATION_AVAILABLE", "total_lines": 20,
        "ranges": [{"start_line": 1, "end_line": 1, "trailing_newline": True},
                   {"start_line": 20, "end_line": 20, "trailing_newline": False}]}
    window = ("[Current source: pkg/a.py]\n[Model source window]\n" + json.dumps(metadata) +
        "\n[Source window content]\n--- source lines 1-1 ---\none\n"
        "--- source lines 20-20 ---\ntwenty\n[End source window]")
    body = {"messages": [{"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": window},
        {"role": "user", "content": "opaque suffix"}]}
    snap = freeze_preparation(body, SimpleNamespace(), request_id="closure", byte_cap=100_000)
    assert [s.payload for s in snap.sources] == ["one\n", "twenty"]
    assert [(s.identity.first_line, s.identity.last_line) for s in snap.sources] == [(1, 1), (20, 20)]
    assert all("[End source window]" not in s.payload for s in snap.sources)
    assert window not in [m.get("content") for m in json.loads(snap.fixed_body)["messages"]]
    assert all(s.payload_start is not None for s in snap.sources)
    from kernel_request_source_window import parse_model_source_window, verify_model_source_window
    for slot in snap.sources:
        selected = parse_model_source_window(slot.page)
        assert selected["continuation"]["omitted_lines"] == 19
        assert verify_model_source_window(slot.page, path="pkg/a.py")


def test_traceback_protection_uses_newest_failing_check():
    rt = SimpleNamespace(_check_output_receipts={
        "old": {"exit_code": 1, "stdout": 'File "old.py", line 1'},
        "new": {"exit_code": 1, "stdout": 'File "pkg/new.py", line 12'},
        "pass": {"exit_code": 0, "stdout": "other.py:3"}})
    body = {"messages": [{"role": "user", "content": "Task: fix"}]}
    snap = freeze_preparation(body, rt, request_id="traceback", byte_cap=100_000)
    assert "pkg/new.py" in snap.protected_paths
    assert "old.py" not in snap.protected_paths
    assert "other.py" not in snap.protected_paths
