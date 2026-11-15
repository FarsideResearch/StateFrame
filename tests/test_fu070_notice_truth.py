"""FU-070-ALLOC: the deferral notice never says a fully carried source is "not attached"."""
from __future__ import annotations

from kernel_request_source_window import DEFERRED_SOURCE_NOTICE_HEADER
from kernel_source_view import file_floor_messages

PATH = "owned_source.py"
SOURCE = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(2100))
TASK = {"role": "user", "content": "Task: inspect the owned source."}
STALE_NOTICE = {"role": "user", "content": (
    f"{DEFERRED_SOURCE_NOTICE_HEADER}\nSource ranges delivered in this request:\n- none\n"
    f"Required source paths are not attached; read one only if your edit needs it:\n- {PATH}"),
    "_request_budget_section": "stage_evidence"}


def _reconcile(messages: list[dict], *, allocator: str) -> list[dict]:
    from kernel_stateframe_framefit5 import _reconcile_framefit5_source_notices
    import os

    previous = os.environ.get("STATEFRAME_WORKING_SET_ALLOCATOR")
    os.environ["STATEFRAME_WORKING_SET_ALLOCATOR"] = allocator
    try:
        _reconcile_framefit5_source_notices(messages, {}, required_deferred_paths=(PATH,))
    finally:
        if previous is None:
            os.environ.pop("STATEFRAME_WORKING_SET_ALLOCATOR", None)
        else:
            os.environ["STATEFRAME_WORKING_SET_ALLOCATOR"] = previous
    return messages


def _texts(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if isinstance(m.get("content"), str)]


def _unattached_mentions(messages: list[dict]) -> list[str]:
    return [text for text in _texts(messages) if "are not attached" in text and PATH in text]


def test_allocator_on_full_pages_and_map_omit_the_notice() -> None:
    messages = _reconcile([TASK, STALE_NOTICE, *file_floor_messages(PATH, SOURCE)], allocator="on")
    assert _unattached_mentions(messages) == []
    assert not any(text.startswith(DEFERRED_SOURCE_NOTICE_HEADER) for text in _texts(messages))
    assert not any("Other required source paths" in text for text in _texts(messages))


def test_allocator_on_revisionless_raw_block_keeps_the_notice() -> None:
    # A raw block carries no revision, so its freshness cannot be proven without bytes.
    pages = file_floor_messages(PATH, SOURCE)
    raw = {"role": "user", "content": f"[Current source: {PATH}]\n{SOURCE}"}
    messages = _reconcile([TASK, STALE_NOTICE, raw, pages[-1]], allocator="on")
    assert _unattached_mentions(messages)


def test_allocator_on_partial_pages_still_list_the_path() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    # Page 1 and the map only: the rest of the file is genuinely absent and must stay deferred.
    messages = _reconcile([TASK, STALE_NOTICE, pages[0], pages[3]], allocator="on")
    notices = [text for text in _texts(messages) if text.startswith(DEFERRED_SOURCE_NOTICE_HEADER)]
    assert len(notices) == 1
    assert notices[0].rstrip().endswith(f"- {PATH}")


def test_allocator_on_without_a_map_keeps_the_path_listed() -> None:
    pages = file_floor_messages(PATH, SOURCE)
    messages = _reconcile([TASK, STALE_NOTICE, *pages[:3]], allocator="on")
    assert _unattached_mentions(messages)


def test_allocator_off_keeps_the_legacy_notice_even_when_the_source_is_carried() -> None:
    messages = _reconcile([TASK, STALE_NOTICE, *file_floor_messages(PATH, SOURCE)], allocator="off")
    assert _unattached_mentions(messages)


def test_allocator_on_stale_revision_pages_do_not_satisfy_the_current_map() -> None:
    stale_source = "".join(f"OLD_{line:04d} = {line}\n" for line in range(2100))
    current_map = file_floor_messages(PATH, SOURCE)[-1]
    stale_pages = file_floor_messages(PATH, stale_source)[:-1]
    messages = _reconcile([TASK, STALE_NOTICE, *stale_pages, current_map], allocator="on")
    assert _unattached_mentions(messages)


def test_allocator_off_keeps_the_legacy_notice_text_byte_for_byte() -> None:
    # Allocator off: the legacy "not attached" notice and host fact must be unchanged.
    off = _reconcile([TASK, STALE_NOTICE, *file_floor_messages(PATH, SOURCE)], allocator="off")
    notices = [text for text in _texts(off) if text.startswith("[Host fact]")]
    assert notices and f"Other required source paths are not attached; read one only if your edit needs it: {PATH}" in notices[0]
    deferred = [text for text in _texts(off) if text.startswith(DEFERRED_SOURCE_NOTICE_HEADER)]
    assert deferred and deferred[0].endswith(f"- {PATH}")
