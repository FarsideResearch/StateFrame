"""edit_source executor: typed refusals, one publish, replay returns the stored outcome."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
from kernel_edit_refusal_codes import RefusalCode
from kernel_edit_source_tool import (
    EditOutcome, EditProtocolRunState, covered_spans, execute_edit_source)

SOURCE = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"


def _repo(tmp_path: Path, text: str = SOURCE, name: str = "pkg/mod.py") -> tuple[Path, Path]:
    root = tmp_path / "repo"
    path = root / name
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode("utf-8"))
    return root, path


def _state(path_rel: str, text: str, *, ordinal: int = 7, start=None, end=None,
           kind=SpanKind.SPAN) -> EditProtocolRunState:
    span = DeliveredSpan.from_source(path_rel, text, start_line=start, end_line=end, kind=kind)
    state = EditProtocolRunState()
    state.set_request(mint_handles(ordinal, [span]))
    return state


def _run(root, state, edits, **kw):
    reads = []

    def read_bytes(p: Path) -> bytes:
        reads.append(p)
        return p.read_bytes()
    out = execute_edit_source(root, {"edits": edits}, state=state, read_bytes=read_bytes, **kw)
    return out, reads


def test_clean_batch_mutates_once_and_preserves_other_bytes(tmp_path):
    root, path = _repo(tmp_path)
    state = _state("pkg/mod.py", SOURCE)
    handle = state.handle_map.records[0].handle
    out, _ = _run(root, state, [
        {"handle": handle, "old_text": "return 1", "new_text": "return 10"},
        {"handle": handle, "old_text": "return 2", "new_text": "return 20"},
    ])
    assert out.status == "APPLIED" and out.changed_paths == ("pkg/mod.py",)
    assert out.generation_id
    assert path.read_text() == SOURCE.replace("return 1", "return 10").replace("return 2", "return 20")
    body = json.loads(out.observation())
    assert body["status"] == "APPLIED" and body["refusals"] == []


def test_refusal_reports_every_defect_and_writes_nothing(tmp_path):
    root, path = _repo(tmp_path)
    state = _state("pkg/mod.py", SOURCE)
    handle = state.handle_map.records[0].handle
    before = path.read_bytes()
    out, _ = _run(root, state, [
        {"handle": handle, "old_text": "", "new_text": "x"},
        {"handle": handle, "old_text": "return 9", "new_text": "y"},
        {"handle": "q7h99", "old_text": "return 1", "new_text": "z"},
    ])
    assert out.status == "REJECTED" and out.changed_paths == ()
    assert out.codes() == ("EMPTY_OLD_TEXT", "OLD_NOT_FOUND", "STALE_HANDLE")
    assert path.read_bytes() == before
    assert out.generation_id is None


def test_no_handle_map_refuses_with_stale_handle_only(tmp_path):
    root, path = _repo(tmp_path)
    out, reads = _run(root, EditProtocolRunState(),
                      [{"handle": "q1h1", "old_text": "return 1", "new_text": "x"}])
    assert out.status == "REJECTED" and out.codes() == ("STALE_HANDLE",)
    assert reads == []  # nothing resolved, nothing read


def test_source_changed_between_delivery_and_edit(tmp_path):
    root, path = _repo(tmp_path)
    state = _state("pkg/mod.py", SOURCE)
    handle = state.handle_map.records[0].handle
    path.write_text(SOURCE + "# moved\n")
    out, _ = _run(root, state, [{"handle": handle, "old_text": "return 1", "new_text": "x"}])
    assert out.codes() == ("SOURCE_CHANGED",)
    assert path.read_text() == SOURCE + "# moved\n"


def test_parse_break_is_refused_and_file_untouched(tmp_path):
    root, path = _repo(tmp_path)
    state = _state("pkg/mod.py", SOURCE)
    handle = state.handle_map.records[0].handle
    out, _ = _run(root, state, [{"handle": handle, "old_text": "def b():", "new_text": "def b(:"}])
    assert out.codes() == ("PARSE_BREAK",)
    assert path.read_text() == SOURCE


def test_not_editable_hook_refuses_before_reading(tmp_path):
    root, path = _repo(tmp_path)
    state = _state("pkg/mod.py", SOURCE)
    handle = state.handle_map.records[0].handle
    out, reads = _run(root, state, [{"handle": handle, "old_text": "return 1", "new_text": "x"}],
                      not_editable=lambda rel: "protected public test paths are immutable")
    assert out.codes() == ("NOT_EDITABLE",) and reads == []
    assert "immutable" in out.detail


def test_redelivery_of_the_same_request_replays_the_stored_outcome(tmp_path):
    root, path = _repo(tmp_path)
    state = _state("pkg/mod.py", SOURCE)
    handle = state.handle_map.records[0].handle
    edits = [{"handle": handle, "old_text": "return 1", "new_text": "return 10"}]
    first, _ = _run(root, state, edits)
    assert first.status == "APPLIED"
    after_first = path.read_bytes()
    second, reads = _run(root, state, edits)
    assert second.status == "REPLAYED" and reads == []
    assert second.generation_id == first.generation_id
    assert second.changed_paths == first.changed_paths
    assert path.read_bytes() == after_first  # the second delivery mutated nothing


def test_idempotent_key_is_deterministic_from_ordinal_and_batch():
    a = _state("pkg/mod.py", SOURCE, ordinal=3)
    b = _state("pkg/mod.py", SOURCE, ordinal=3)
    c = _state("pkg/mod.py", SOURCE, ordinal=4)
    args = {"edits": [{"handle": "q3h1", "old_text": "x", "new_text": "y"}]}
    assert a.idempotent_key("edit_source", args) == b.idempotent_key("edit_source", args)
    assert a.idempotent_key("edit_source", args) != c.idempotent_key("edit_source", args)


def test_crlf_and_bom_bytes_survive_an_applied_edit(tmp_path):
    text = "﻿def a():\r\n    return 1\r\n"
    root, path = _repo(tmp_path, text, name="pkg/win.py")
    state = _state("pkg/win.py", text)
    handle = state.handle_map.records[0].handle
    out, _ = _run(root, state, [{"handle": handle, "old_text": "return 1", "new_text": "return 2"}])
    assert out.status == "APPLIED"
    assert path.read_bytes() == text.replace("return 1", "return 2").encode("utf-8")


def test_covered_spans_mirror_handle_records():
    state = _state("pkg/mod.py", SOURCE, start=5, end=6)
    (span,) = covered_spans(state.handle_map)
    record = state.handle_map.records[0]
    assert (span.handle, span.path, span.revision, span.start, span.end, span.kind) == (
        record.handle, record.path, record.revision, record.span.byte_start, record.span.byte_end, "SPAN")
    assert covered_spans(None) == ()


def test_malformed_edits_are_rejected_without_a_protocol_code(tmp_path):
    root, _ = _repo(tmp_path)
    out = execute_edit_source(root, {"edits": "nope"}, state=EditProtocolRunState(),
                              read_bytes=Path.read_bytes)
    assert out.status == "REJECTED" and out.refusals == () and out.detail.startswith("edit_source REJECTED:")
