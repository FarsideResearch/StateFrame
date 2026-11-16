"""FU-070-ALLOC off-parity: allocator-only read/edit shapes stay gated behind the switch.

With STATEFRAME_WORKING_SET_ALLOCATOR off, the read-window clip set and the fallback
edit locator must match the pre-allocator behavior exactly.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from kernel_source_view import file_floor_messages
from stateframe.engine.kernel_stateframe_action_ledger import (
    _fallback_edit_open_site_locator,
    _read_site_window,
)

SWITCH = "STATEFRAME_WORKING_SET_ALLOCATOR"
PATH = "owned_source.py"
SOURCE = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(2100))
TASK = {"role": "user", "content": "Task: inspect the owned source."}


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _clipped_read(revision: str, clip: str, delivered: int, total: int) -> str:
    """A read page clipped by ``clip`` with the effective range in lines= and truncated=true."""
    body = "".join(f"VALUE_{i:04d} = {i}\n" for i in range(delivered))
    header = (
        f"[read page] path={PATH} revision={revision} "
        f"lines=1-{delivered}/{total} requested=lines:1-1000 "
        f"effective=lines:1-{delivered} clip={clip} truncated=true "
        f"chars={len(body)} bytes={len(body.encode('utf-8'))} "
        "decoded_with_replacement=false"
    )
    return f"{header}\nmore content\n{body}"


def _read_args() -> dict[str, object]:
    return {"path": PATH, "start_line": 1, "end_line": 1000}


def test_max_chars_clipped_page_is_refused_with_allocator_off(monkeypatch) -> None:
    monkeypatch.delenv(SWITCH, raising=False)
    revision = _sha(SOURCE.encode("utf-8"))
    result = _clipped_read(revision, "max_chars", 683, 2100)
    assert _read_site_window(_read_args(), result) is None


def test_max_chars_clipped_page_is_accepted_with_allocator_on(monkeypatch) -> None:
    monkeypatch.setenv(SWITCH, "on")
    revision = _sha(SOURCE.encode("utf-8"))
    result = _clipped_read(revision, "max_chars", 683, 2100)
    parsed = _read_site_window(_read_args(), result)
    assert parsed is not None
    fact, _content = parsed
    assert fact["lines"] == {"start": 1, "end": 683, "total": 2100}


def test_max_lines_clipped_page_is_still_accepted_with_allocator_off(monkeypatch) -> None:
    # The historical clip set is {none, max_lines}; that path must not regress with the switch off.
    monkeypatch.delenv(SWITCH, raising=False)
    revision = _sha(SOURCE.encode("utf-8"))
    result = _clipped_read(revision, "max_lines", 683, 2100)
    assert _read_site_window(_read_args(), result) is not None


def test_fallback_edit_without_prior_read_anchors_line_one_with_allocator_off(monkeypatch) -> None:
    monkeypatch.delenv(SWITCH, raising=False)
    args = {"path": PATH, "old_str": "VALUE = 1\n", "new_str": "VALUE = 2\n"}
    locator = _fallback_edit_open_site_locator(
        "str_replace", args, "str_replace applied.", {}, _sha(b"post-edit source")
    )
    assert locator is not None
    assert locator["path"] == PATH
    assert locator["lines"]["start"] == 1
    assert locator["last_read_revision"] == ""


def test_fallback_edit_without_prior_read_is_none_with_allocator_on(monkeypatch) -> None:
    monkeypatch.setenv(SWITCH, "on")
    args = {"path": PATH, "old_str": "VALUE = 1\n", "new_str": "VALUE = 2\n"}
    locator = _fallback_edit_open_site_locator(
        "str_replace", args, "str_replace applied.", {}, _sha(b"post-edit source")
    )
    assert locator is None


def test_admit_sent_edit_without_capture_fails_closed(monkeypatch) -> None:
    # prepared.capture is None: no AttributeError, and the empty request id is refused.
    from probes import react_repair

    pages = file_floor_messages(PATH, SOURCE)
    monkeypatch.setattr(react_repair, "_sent_request_messages", lambda prepared: [TASK, *pages])
    prepared = SimpleNamespace(messages=(TASK,), capture=None)
    ok, reason, _detail = react_repair._admit_sent_edit(
        prepared, path=PATH, source=SOURCE.encode("utf-8")
    )
    assert ok is False
    assert reason == "no_request_id"
