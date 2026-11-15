"""FU-070-ALLOC: the child edit gate judges the request that was sent, not the pre-render list."""
from __future__ import annotations

from types import SimpleNamespace

from kernel_source_view import file_floor_messages

PATH = "owned_source.py"
SOURCE = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(2100))
TASK = {"role": "user", "content": "Task: inspect the owned source."}


def _prepared(pre_render: list[dict], sent: list[dict] | None) -> SimpleNamespace:
    """A prepared request whose pre-render list lacks the source window; ``sent`` is the wire."""
    qualified = None if sent is None else SimpleNamespace(body={"messages": sent})
    capture = SimpleNamespace(request_id="req-sent", qualified_request=qualified)
    return SimpleNamespace(messages=tuple(pre_render), capture=capture)


def _admit(prepared: SimpleNamespace):
    from probes.react_repair import _admit_sent_edit

    return _admit_sent_edit(prepared, path=PATH, source=SOURCE.encode("utf-8"))


def test_edit_admitted_from_complete_sent_wire_when_pre_render_has_no_pages() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    prepared = _prepared([TASK], [TASK, *pages])
    assert _admit(prepared) == (True, "", "")


def test_edit_refused_when_sent_wire_is_missing_a_middle_page() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    # pages[0..2] are the three source pages and pages[3] is the map line; drop the middle page.
    partial = [TASK, pages[0], pages[2], pages[3]]
    ok, reason, _detail = _admit(_prepared([TASK], partial))
    assert ok is False
    assert reason == "incomplete_view"


def test_edit_refused_when_only_one_page_of_several_is_sent() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    ok, reason, _detail = _admit(_prepared([TASK], [TASK, pages[0], pages[3]]))
    assert ok is False
    assert reason == "incomplete_view"


def test_edit_fails_closed_without_a_qualified_sent_wire() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    # The pre-render list is complete, but no sent wire exists: must refuse, never admit on pre-render.
    ok, reason, _detail = _admit(_prepared([TASK, *pages], None))
    assert ok is False
    assert reason == "incomplete_view"
