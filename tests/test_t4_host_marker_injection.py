"""Host-marker injection: model, tool, file and task text carrying every host marker must never become host rows.

Incidents this guards: a reply opening with a notice header was treated as a host row; a tool result quoting
"[CONTINUATION LIMITS |" made the final cache authority non-unique; a "[Current source:" row the model wrote
crashed _window_entries. The fixture takes the marker set from the engine sources, so a new marker joins it
without an edit here.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from _boundary_inputs import BOUNDARY_TEXTS, call_bounded
from _host_markers import (
    KNOWN_MARKERS,
    host_markers,
    injected_text,
    lookalike_window,
    marker_lines,
    scan_engine_markers,
)
from enginecheck import HarnessError, run
from enginecheck.boundary import assert_survives, read_scenario

MARKERS = host_markers()
_HOST_PATH = "target.py"
_LOOKALIKE_PATH = "injected/path.py"


# --- marker collection -------------------------------------------------------------------------------------------

def test_scan_finds_every_pinned_marker() -> None:
    scanned = scan_engine_markers()
    missing = sorted(KNOWN_MARKERS - scanned)
    assert not missing, f"engine sources no longer carry pinned host markers: {missing}"


def test_scan_rejects_regex_character_classes() -> None:
    assert not [marker for marker in MARKERS if marker in {"[A-Za-z_]", "[A-Z]", "[A-Za-z0-9_]"}]
    assert all(marker.startswith("[") for marker in MARKERS)


def test_injected_text_carries_every_marker() -> None:
    text = injected_text("probe")
    for marker in MARKERS:
        assert marker in text, marker
    assert lookalike_window() in text


# --- engine-level: a model-free run survives injected markers everywhere ------------------------------------------

# The first test puts only this one marker line in the file, so a failure points at file-side marker handling.
# The second test carries every marker (about 5.7 KB) in the file itself.
_FILE_SAFE_MARKERS = ("[Current source:",)

# Separate engine limit, not the evidence pin: a read of a ~6 KB file suspends the run before the done step on the
# r2hc3 profile, with or without markers (BYTE_POLICY_DOMAIN_EXHAUSTED floor 97693 > cap 82026, or
# "frame-only typed state could not be constructed"). Remove this xfail when the large-read budget is fixed.
_LARGE_READ_REASON = ("TESTGAP-T4-LARGE-READ-BUDGET: ~6 KB read suspends before done on r2hc3 "
                      "(BYTE_POLICY_DOMAIN_EXHAUSTED floor 97693 > cap 82026); marker-free files hit it too")


def _injection_scenario(content: bytes):
    scenario = read_scenario("host_marker_injection", content, path=_HOST_PATH)
    task = "Task text with markers:\n" + marker_lines() + f"Read {_HOST_PATH}, then update notes.md."
    steps = [
        {"tool": "read", "required": True, "arguments": {"path": _HOST_PATH}},
        {"tool": "done", "required": True,
         "arguments": {"summary": "Model summary with markers:\n" + injected_text("summary")}},
    ]
    return replace(scenario, task=task, steps=steps)


def _host_source_rows(request: dict) -> list[str]:
    """User rows that open with the current-source header: only the host writes those."""
    return [row["content"] for row in request.get("messages", [])
            if isinstance(row, dict) and row.get("role") == "user"
            and isinstance(row.get("content"), str) and row["content"].startswith("[Current source:")]


def test_engine_survives_markers_in_task_tool_reply_and_file(tmp_path: Path) -> None:
    content = (marker_lines(_FILE_SAFE_MARKERS) + "a = 1\n").encode("utf-8")
    scenario = _injection_scenario(content)
    trace = run(scenario, tmp_path)

    assert_survives(trace, scenario)
    assert "final cache authority" not in trace.output
    for artifact in trace.run_dir.rglob("*"):
        if artifact.is_file() and "wire" not in artifact.relative_to(trace.run_dir).parts:
            assert b"final cache authority" not in artifact.read_bytes(), artifact
    for request in trace.request_json:
        for row_content in _host_source_rows(request):
            assert not row_content.startswith(f"[Current source: {_LOOKALIKE_PATH}]"), row_content[:120]


@pytest.mark.xfail(strict=True, raises=HarnessError, reason=_LARGE_READ_REASON)
def test_engine_survives_every_marker_as_file_content(tmp_path: Path) -> None:
    scenario = _injection_scenario(injected_text("file content").encode("utf-8"))
    trace = run(scenario, tmp_path)
    assert_survives(trace, scenario)


# --- unit-level: window and row parsers keep model-supplied markers out of the host inventory ---------------------

def _ownership(prefix_end: int, frame_index: int, exchange_start: int):
    from kernel_stable_wire_ownership import RequestOwnership
    return RequestOwnership(prefix_end, frame_index, exchange_start, exchange_start, exchange_start)


def _layout(model_window: str) -> tuple[list[dict], object]:
    """system, task, one host window, frame, then an exchange whose rows all carry the model's lookalike."""
    host_window = lookalike_window(_HOST_PATH, revision="sha256:" + "1" * 64)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: read"},
        {"role": "user", "content": host_window},
        {"role": "user", "content": "[StateFrame]\n{}"},
        {"role": "assistant", "content": model_window},
        {"role": "tool", "tool_call_id": "call_1", "content": model_window},
        {"role": "user", "content": model_window},
    ]
    return messages, _ownership(2, 3, 4)


def test_window_entries_keep_only_the_host_window() -> None:
    from kernel_stable_wire import _window_entries
    messages, ownership = _layout(lookalike_window(_LOOKALIKE_PATH))

    entries = _window_entries(messages, ownership)

    assert [entry["identity"]["path"] for entry in entries] == [_HOST_PATH]


def test_window_entries_ignore_a_model_copy_of_the_host_window() -> None:
    """A model row repeating the host window byte-for-byte must not trip the duplicate-window check."""
    from kernel_stable_wire import _window_entries
    messages, ownership = _layout(lookalike_window(_HOST_PATH, revision="sha256:" + "1" * 64))

    entries = _window_entries(messages, ownership)

    assert len(entries) == 1


def test_window_entries_without_ownership_see_no_windows() -> None:
    from kernel_stable_wire import _window_entries
    messages, _ownership_unused = _layout(lookalike_window(_LOOKALIKE_PATH))

    assert _window_entries(messages, None) == []


@pytest.mark.parametrize("marker", MARKERS)
def test_window_entries_are_bounded_on_marker_rows(marker: str) -> None:
    """Every marker as the opening of host and model rows: no host window, so no entries and no error."""
    from kernel_stable_wire import StableWireError, _window_entries
    text = marker + " probe]\n" + marker_lines()
    messages, ownership = _layout(text)
    messages[2] = {"role": "user", "content": text}

    outcome = call_bounded(_window_entries, messages, ownership, allowed=(StableWireError,))

    assert outcome.raised is None, f"{type(outcome.raised).__name__}: {outcome.raised}"
    assert outcome.value == []


@pytest.mark.parametrize(("name", "text"), sorted(BOUNDARY_TEXTS.items()))
def test_window_entries_are_bounded_on_marker_plus_boundary_text(name: str, text: str) -> None:
    from kernel_stable_wire import StableWireError, _window_entries
    content = "[Current source: " + text + "]\n[Model source window]\n" + text
    messages, ownership = _layout(content)
    messages[2] = {"role": "user", "content": content}

    outcome = call_bounded(_window_entries, messages, ownership, allowed=(StableWireError,))

    assert outcome.raised is None, f"{type(outcome.raised).__name__}: {outcome.raised}"
    assert outcome.value == []


def test_source_window_order_key_is_bounded() -> None:
    """Marker rows with no window body have no order key. The control row proves the helper can return a key,
    so a None-returning stub cannot satisfy this test."""
    from kernel_cache_layout_candidate import _source_window_order_key
    assert _source_window_order_key(lookalike_window(_LOOKALIKE_PATH)) is not None
    for marker in MARKERS:
        for content in (marker, marker + "\n" + marker_lines(), injected_text(marker)):
            outcome = call_bounded(_source_window_order_key, content)
            assert outcome.raised is None, marker
            assert outcome.value is None, (marker, outcome.value)


def test_source_window_order_key_reads_a_lookalike_only_by_shape() -> None:
    """The key helper has no ownership input; callers must gate it. A lookalike keys by its path and line starts."""
    from kernel_cache_layout_candidate import _source_window_order_key
    outcome = call_bounded(_source_window_order_key, lookalike_window(_LOOKALIKE_PATH))
    assert outcome.raised is None
    assert outcome.value == '{"path":"injected/path.py","starts":[1]}'


@pytest.mark.parametrize("role", ["assistant", "tool", "user"])
def test_reservation_kind_is_bounded_on_marker_rows(role: str) -> None:
    from kernel_request_budget import _reservation_kind
    # Host-authored rows (system/user) reserve as required source; assistant and tool rows are inert data
    # and never reserve (kernel_request_budget._HOST_SOURCE_ROLES).
    expected = "required_source" if role in ("system", "user") else None
    for marker in MARKERS:
        outcome = call_bounded(_reservation_kind, {"role": role, "content": injected_text(marker)})
        assert outcome.raised is None, marker
        assert outcome.value == expected, (marker, role, outcome.value)


@pytest.mark.parametrize("role", ["assistant", "tool", "user"])
def test_pinned_render_check_is_bounded_on_marker_rows(role: str) -> None:
    from kernel_stateframe_runtime import _is_pinned_render_message
    for marker in MARKERS:
        outcome = call_bounded(_is_pinned_render_message, {"role": role, "content": injected_text(marker)}, 5, set())
        assert isinstance(outcome.value, bool), marker


def test_source_body_is_bounded_on_marker_text() -> None:
    from kernel_real04_bugs import _source_body
    for marker in MARKERS:
        assert isinstance(call_bounded(_source_body, injected_text(marker)).value, str), marker


def test_fit_requested_message_is_bounded_on_marker_text() -> None:
    from kernel_stateframe_runtime import _fit_requested_message
    for marker in MARKERS:
        message = {"role": "user", "content": injected_text(marker)}
        assert call_bounded(_fit_requested_message, message).raised is None, marker
        assert isinstance(message.get("content"), str), marker


def test_compact_messages_are_bounded_on_marker_rows() -> None:
    from kernel_v33_lean import compact_messages
    for marker in MARKERS:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Task: read"},
            {"role": "assistant", "content": injected_text(marker)},
            {"role": "tool", "tool_call_id": "call_1", "content": injected_text(marker)},
        ]
        outcome = call_bounded(compact_messages, messages, allowed=(ValueError,))
        assert outcome.raised is None, (marker, outcome.raised)
        assert isinstance(outcome.value, list), marker
        assert [row["role"] for row in outcome.value] == ["system", "user", "assistant", "tool"], marker
