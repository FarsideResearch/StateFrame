"""FU-070-ALLOC post-edit pass 2: exact site windows and post-edit fact revisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from stateframe.engine.kernel_stateframe_action_ledger import (
    _read_site_window,
    build_action_context,
)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _source(lines: int) -> bytes:
    return "".join(f"VALUE_{i:04d} = {i}\n" for i in range(lines)).encode("utf-8")


def _clipped_read(revision: str, delivered: int, total: int) -> str:
    """Build a read result as the engine emits it when max_chars clips the page."""
    body = "".join(f"VALUE_{i:04d} = {i}\n" for i in range(delivered))
    header = (
        f"[read page] path=owned_source.py revision={revision} "
        f"lines=1-{delivered}/{total} requested=lines:1-1000 "
        f"effective=lines:1-{delivered} clip=max_chars truncated=true "
        f"chars={len(body)} bytes={len(body.encode('utf-8'))} "
        "decoded_with_replacement=false"
    )
    return f"{header}\nmore content\n{body}"


def _read_call(call_id: str, arguments: dict[str, object], result: str) -> list[dict[str, object]]:
    return [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function",
             "function": {"name": "read", "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def test_max_chars_clipped_page_is_an_exact_site_at_its_effective_range(monkeypatch) -> None:
    # D1: the delivered bytes verify, so the clipped page is an exact window for lines 1-683.
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    revision = _sha(_source(2100))
    result = _clipped_read(revision, 683, 2100)
    parsed = _read_site_window({"path": "owned_source.py", "start_line": 1, "end_line": 1000}, result)
    assert parsed is not None
    fact, content = parsed
    assert fact["lines"] == {"start": 1, "end": 683, "total": 2100}
    assert fact["revision"] == revision
    assert content.count("\n") == 683


def test_max_chars_clipped_page_with_wrong_declared_bytes_is_refused(monkeypatch) -> None:
    # D1 keeps refusing any clipped page whose bytes do not verify.
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    revision = _sha(_source(2100))
    result = _clipped_read(revision, 683, 2100).replace("bytes=", "bytes=1", 1)
    assert _read_site_window({"path": "owned_source.py"}, result) is None


def test_max_chars_clipped_read_yields_open_edit_site(tmp_path: Path, monkeypatch) -> None:
    # D1 end-to-end: the ledger builds an exact current site from the clipped page.
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    source = _source(2100)
    (tmp_path / "owned_source.py").write_bytes(source)
    revision = _sha(source)
    messages: list[dict[str, object]] = [{"role": "user", "content": "Task: inspect source"}]
    messages += _read_call(
        "c1", {"path": "owned_source.py", "start_line": 1, "end_line": 1000},
        _clipped_read(revision, 683, 2100),
    )
    context = build_action_context(messages, workspace_root=tmp_path)
    assert context.get("open_edit_site") is not None
