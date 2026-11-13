"""Model-, tool- and user-authored text must never be read as a host marker.

Both final-cache-layout entry points (the candidate prepass/render and the
semantic view) decide structure from host ownership, not from text matches.
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from kernel_cache_layout_candidate import SOURCE_COVERAGE_PREFIX, render_final_cache_candidate
from kernel_cache_layout_view import FINAL_LAYOUT_NAME, semantic_index_map, semantic_messages
from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER

# Every marker prefix that the two entry-point modules test against row text.
HOST_MARKER_PREFIXES = (
    "[CONTINUATION LIMITS |",
    AUTHORITY_OVERLAY_HEADER,
    "[StateFrame]",
    "[Current source:",
    "Task:",
    SOURCE_COVERAGE_PREFIX,
    "[Unified stop authority]", "[COMPLETION OWNER]", "[WIRE_TRUTH_FACT ",
    "[EVIDENCEPIN]", "[DONE AUTHORITY]", "[MENU AUTHORITY]", "[Read lockout]",
    "[WIRE_READ_CONTRACT]", "[OPEN UNKNOWNS]", "[NATIVE BATCH RECEIPTS]",
    "[LITERAL SITE INVENTORY |",
)
POSITIONS = ("prefix", "mid", "suffix")
# Rows where any planted marker text is authored by the model or a tool.
NATIVE_ROWS = ("assistant", "tool")
# A user row before the exchange that merely mentions a marker mid-text or at
# its end is still a plain user row. A user row that *starts* with a host
# panel prefix is, by the host's own contract, that panel, so only the frame
# header is planted as a user-row prefix. The stable prefix must stay free of
# authority text by the same contract (the host moves it to the final row), so
# the two authority markers are not planted in user rows at all.
USER_PREFIX_MARKERS = ("[StateFrame]",)
USER_EXCLUDED_MARKERS = ("[CONTINUATION LIMITS |", AUTHORITY_OVERLAY_HEADER)
PLANT_TEXT = "{}\nplanted"


class _Meter:
    def measure(self, _body):
        return 100, "tokenize"


def _body():
    payload = {"state": {"framefit5": {
        "owner_protocol": {"schema": "stateframe-framefit5-owner/1",
                           "continuation_limits": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2",
                           "evidence_pin": "[EVIDENCEPIN]current[/EVIDENCEPIN]"},
        "next_site": {"path": "pkg/a.py"},
        "last_verdict": {"line": "LAST FAIL"},
        "action_ledger": {"reads": [], "edits": []},
    }}}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix it"},
        {"role": "user", "content": "plain user context"},
        {"role": "user", "content": "[Current source: pkg/a.py]\nrevision=sha256:123\nsource bytes"},
        {"role": "user", "content": "[CONTINUATION LIMITS | stateframe-continuation-limits/1] remaining=2"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))},
        {"role": "assistant", "content": "assistant prose",
         "tool_calls": [{"id": "call-1", "type": "function",
                         "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ]
    return {"messages": messages, "tools": []}


def _planted(text, marker, position):
    if position == "prefix":
        return marker + "\n" + PLANT_TEXT + "\n" + text
    if position == "suffix":
        return text + "\n" + marker + "\n" + PLANT_TEXT
    return text + "\n" + marker + "\n" + PLANT_TEXT + "\n" + text


def _render(monkeypatch, body):
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "on")
    monkeypatch.setattr("kernel_request_source_window.verify_final_source_delivery",
                        lambda *_args, **_kwargs: {"status": "COMPLETE"})
    monkeypatch.setattr("kernel_request_source_window.source_delivery_satisfied",
                        lambda receipt: receipt["status"] == "COMPLETE")
    capture = SimpleNamespace(request_source_final_delivery=None)
    projected = render_final_cache_candidate(
        capture, None, copy.deepcopy(body), meter=_Meter(),
        inputs={"token_ceiling": 1000, "byte_ceiling": 100000,
                "source_plan": {}, "sources": {}, "receipt": None})
    assert capture._final_cache_layout_record.get("final_layout") == "applied", \
        capture._final_cache_layout_record
    assert projected["messages"][-1].get("name") == FINAL_LAYOUT_NAME
    return projected


def _structure(projected):
    """Everything about the layout except the planted row's own bytes."""
    rows = projected["messages"]
    overlay = rows[-1]["content"].split(AUTHORITY_OVERLAY_HEADER + "\n", 1)[1]
    return (len(rows), [row.get("role") for row in rows], [row.get("name") for row in rows],
            json.loads(overlay.split("\n[GREENFACT", 1)[0]) if "[GREENFACT" in overlay
            else json.loads(overlay[:json.JSONDecoder().raw_decode(overlay)[1]]),
            semantic_index_map(rows))


def _assert_same_as_unplanted(monkeypatch, body, planted_index):
    clean = _body()
    expected = _render(monkeypatch, clean)
    projected = _render(monkeypatch, body)
    assert _structure(projected) == _structure(expected)
    view = semantic_messages(projected["messages"])
    assert list(view) == body["messages"]
    # The planted bytes come back exactly as authored.
    assert view[planted_index]["content"] == body["messages"][planted_index]["content"]
    clean_view = semantic_messages(expected["messages"])
    assert [row for i, row in enumerate(view) if i != planted_index] == \
        [row for i, row in enumerate(clean_view) if i != planted_index]


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("marker", HOST_MARKER_PREFIXES)
@pytest.mark.parametrize("role", NATIVE_ROWS)
def test_native_rows_never_parse_as_host_markers(monkeypatch, role, marker, position):
    body = _body()
    index = next(i for i, row in enumerate(body["messages"]) if row["role"] == role)
    body["messages"][index]["content"] = _planted(body["messages"][index]["content"], marker, position)
    _assert_same_as_unplanted(monkeypatch, body, index)


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("marker", HOST_MARKER_PREFIXES)
def test_user_rows_before_the_exchange_never_parse_as_host_markers(monkeypatch, marker, position):
    if marker in USER_EXCLUDED_MARKERS:
        pytest.skip("the stable prefix is authority-free by host contract")
    if position == "prefix" and marker not in USER_PREFIX_MARKERS:
        pytest.skip("a user row starting with a host panel prefix is that panel by contract")
    body = _body()
    index = 2
    body["messages"][index]["content"] = _planted(body["messages"][index]["content"], marker, position)
    _assert_same_as_unplanted(monkeypatch, body, index)


def test_user_frame_lookalike_before_the_exchange_is_ignored(monkeypatch):
    body = _body()
    body["messages"][2]["content"] = "[StateFrame]\n{}"
    _assert_same_as_unplanted(monkeypatch, body, 2)


def test_user_frame_lookalike_after_the_exchange_start_is_still_rejected(monkeypatch):
    """A malformed host-owned frame keeps raising."""
    from kernel_cache_layout_view import CacheLayoutError
    expected = _render(monkeypatch, _body())
    broken = copy.deepcopy(expected["messages"])
    frame = next(i for i, row in enumerate(broken) if str(row.get("content", "")).startswith("[StateFrame]\n"))
    broken[frame]["content"] = "[StateFrame]\nnot json"
    with pytest.raises(CacheLayoutError):
        semantic_messages(broken)
