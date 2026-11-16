"""The subframe edit gate accepts the worker's own current-revision read page as source.

After a subframe reads an owned file, the host retires that file's source window and
the read page in the tool result is the only copy on the wire. The gate must verify it
byte for byte, and must not accept a stale, partial or altered page.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

PATH = "left.py"
SOURCE = b"VALUE = 1\n"
TASK = {"role": "user", "content": "Task: update the left adapter."}


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _page(body: str, *, revision: str | None = None, lines: str = "1-1/1",
          path: str = PATH) -> dict:
    revision = revision or _digest(SOURCE)
    header = (f"[read page] path={path} revision={revision} lines={lines} "
              "requested=lines:1-120 effective=lines:1-1@0:10 clip=none chars=10 bytes=10 "
              "truncated=false decoded_with_replacement=false\n0 more; complete=true\n")
    return {"role": "tool", "tool_call_id": "call_0", "content": header + body}


def _admit(messages: list[dict], source: bytes = SOURCE):
    from probes.react_repair import _admit_sent_edit

    capture = SimpleNamespace(request_id="req-1",
                              qualified_request=SimpleNamespace(body={"messages": messages}))
    prepared = SimpleNamespace(messages=tuple(messages), capture=capture)
    return _admit_sent_edit(prepared, path=PATH, source=source)


def test_full_current_read_page_admits_the_edit() -> None:
    assert _admit([TASK, _page("VALUE = 1\n")]) == (True, "", "")


def test_read_page_with_a_host_suffix_still_admits() -> None:
    assert _admit([TASK, _page("VALUE = 1\n\n[host] end of file\n")])[0] is True


def test_stale_revision_read_page_is_refused() -> None:
    stale = _page("VALUE = 1\n", revision=_digest(b"VALUE = 0\n"))
    assert _admit([TASK, stale])[1] == "incomplete_view"


def test_read_page_whose_body_differs_from_source_is_refused() -> None:
    assert _admit([TASK, _page("VALUE = 2\n")])[1] == "incomplete_view"


def test_partial_read_page_does_not_admit_a_multi_line_file() -> None:
    source = b"A = 1\nB = 2\n"
    page = _page("A = 1\n", revision=_digest(source), lines="1-1/2")
    assert _admit([TASK, page], source=source)[1] == "incomplete_view"


def test_read_page_for_another_path_is_ignored() -> None:
    assert _admit([TASK, _page("VALUE = 1\n", path="right.py")])[1] == "incomplete_view"
