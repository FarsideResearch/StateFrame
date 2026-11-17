"""rewrite_source: REWRITE handle only, destructive-guarded, one publish, replay-stable."""
from __future__ import annotations

from pathlib import Path

import pytest

from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
from kernel_edit_source_tool import EXECUTORS, EditProtocolRunState, execute_rewrite_source
from kernel_edit_tool_names import DESTRUCTIVE_EDIT_TOOLS

SOURCE = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
NEW = "def a():\n    return 100\n\n\ndef b():\n    return 200\n"


def _repo(tmp_path: Path, text: str = SOURCE) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    path = root / "pkg/mod.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(text.encode("utf-8"))
    return root, path


def _state(kind: SpanKind, text: str = SOURCE, ordinal: int = 2) -> EditProtocolRunState:
    state = EditProtocolRunState()
    state.set_request(mint_handles(ordinal, [DeliveredSpan.from_source("pkg/mod.py", text, kind=kind)]))
    return state


def _run(root, state, args, **kw):
    reads = []

    def read_bytes(p):
        reads.append(p)
        return p.read_bytes()
    return execute_rewrite_source(root, args, state=state, read_bytes=read_bytes, **kw), reads


def test_rewrite_with_rewrite_handle_replaces_the_whole_region_once(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_EDIT_SHRINK_GUARD", raising=False)
    root, path = _repo(tmp_path)
    state = _state(SpanKind.REWRITE)
    handle = state.handle_map.records[0].handle
    out, _ = _run(root, state, {"handle": handle, "new_text": NEW})
    assert out.status == "APPLIED" and out.changed_paths == ("pkg/mod.py",)
    assert path.read_text() == NEW
    assert "admission=" in out.detail


def test_span_handle_is_refused_as_not_editable_without_reading(tmp_path):
    root, path = _repo(tmp_path)
    state = _state(SpanKind.SPAN)
    handle = state.handle_map.records[0].handle
    out, reads = _run(root, state, {"handle": handle, "new_text": NEW})
    assert out.codes() == ("NOT_EDITABLE",) and reads == []
    assert path.read_text() == SOURCE


def test_unknown_handle_is_stale(tmp_path):
    root, _ = _repo(tmp_path)
    out, _ = _run(root, _state(SpanKind.REWRITE), {"handle": "q2h9", "new_text": NEW})
    assert out.codes() == ("STALE_HANDLE",)


def test_moved_file_is_source_changed(tmp_path):
    root, path = _repo(tmp_path)
    state = _state(SpanKind.REWRITE)
    handle = state.handle_map.records[0].handle
    path.write_text(SOURCE + "\n# tail\n")
    out, _ = _run(root, state, {"handle": handle, "new_text": NEW})
    assert out.codes() == ("SOURCE_CHANGED",)


def test_parse_break_refused(tmp_path):
    root, path = _repo(tmp_path)
    state = _state(SpanKind.REWRITE)
    handle = state.handle_map.records[0].handle
    out, _ = _run(root, state, {"handle": handle, "new_text": "def a(:\n"})
    assert out.codes() == ("PARSE_BREAK",) and path.read_text() == SOURCE


def test_shrink_guard_blocks_a_destructive_rewrite_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_EDIT_SHRINK_GUARD", "on")
    big = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(40))
    root, path = _repo(tmp_path, big)
    state = _state(SpanKind.REWRITE, big)
    handle = state.handle_map.records[0].handle
    out, _ = _run(root, state, {"handle": handle, "new_text": "def f0():\n    return 0\n"})
    assert out.status == "REJECTED" and out.refusals == ()
    assert "REJECTED_SHRINK" in out.detail
    assert path.read_text() == big


def test_rewrite_is_classified_destructive_and_registered():
    assert "rewrite_source" in DESTRUCTIVE_EDIT_TOOLS
    assert set(EXECUTORS) == {"edit_source", "rewrite_source"}


def test_redelivery_replays_without_touching_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_EDIT_SHRINK_GUARD", raising=False)
    root, path = _repo(tmp_path)
    state = _state(SpanKind.REWRITE)
    handle = state.handle_map.records[0].handle
    args = {"handle": handle, "new_text": NEW}
    first, _ = _run(root, state, args)
    second, reads = _run(root, state, args)
    assert first.status == "APPLIED" and second.status == "REPLAYED" and reads == []
    assert path.read_text() == NEW
