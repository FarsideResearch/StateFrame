"""Edit protocol: request-scoped handles minted from delivered source, map in the final user message.

Design (STATEFRAME_EDIT_PROTOCOL=handles; legacy_safe is untouched)
-------------------------------------------------------------------
Owner of the final user message. Every provider body passes through the transport
(RequestCapture send). Both of its branches already have one send-time seam that
appends a host message after every wire note and before visibility, budget,
capacity and seal: the read-once final message (legacy branch in kernel_measurement,
qualified FrameFit5 branch in kernel_final_wire_allocation.build_qualified_request).
The map is placed at that seam by kernel_edit_request_map.apply, so its bytes are
fitted, capacity-checked and sealed with the rest of the wire. Nothing upstream
(react loop transcript, prepare, frame, open sites) ever holds the map.

One minter, two providers (kernel_edit_handles.request_spans):
  A. frame / allocator: open-site rows of the request's [StateFrame] block, read
     through spans_from_open_sites (rows are never written; a stale row is skipped).
  B. legacy source blocks: maximal runs of current file lines the request text
     carries verbatim (source windows, read pages, exact clipped reads). A run must
     hold at least two non-blank lines or be the whole file; gutter-rendered lines
     never match, so a handle never names bytes the model did not see byte-for-byte.
  Spans are unioned per file and minted q{ordinal}h{n} in first-mention order; a run
  covering the whole file is REWRITE (licenses rewrite_source). Paged descriptors and
  allocator source-range handles remain paging addresses, never edit handles.

Request ordinal = the capture's step, the same counter the repair request uses
(step + 1 for the request after a refused batch). Same body + ordinal -> same map,
so retries, capacity refits and replay re-mint identical handles.

Placement: appended to the final message when it is user text, else a NEW user
message after it (tool replies stay verbatim). When the current turn (messages after
the last assistant turn) already carries a map, it is a repair request: no second
map, the bound handles stay. The map binds onto the run's EditProtocolRunState,
which the host attaches to the runtime only under handles; edit_source on request
N's handles therefore resolves on the first call instead of STALE_HANDLE.

Cache: request N+1 is built from a transcript that never held N's map, so N and N+1
share every byte up to where N's map began; the map is never inside a retained
chunk. Site inventory: the frame message and its open-site rows are byte-identical
before and after placement, so site identity (the ed2f512e exclusion set) is
unchanged across N and N+1 without a new carve-out.

Failure: any mint error is fail-open: body unchanged, bound map cleared, and the
executor's STALE_HANDLE refusal delivers a repair request with fresh handles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import kernel_edit_request_map as request_map  # noqa: E402
from kernel_edit_handles import (  # noqa: E402
    CURRENT_SOURCE_PREFIX, SpanKind, compose_request_map, current_turn_has_map, parse_map,
    request_spans, verbatim_line_runs,
)
from kernel_edit_source_tool import APPLIED, REJECTED, EditProtocolRunState, execute_edit_source  # noqa: E402
from kernel_frame_optional_allocation import site_identity  # noqa: E402
from kernel_paged_source import source_revision  # noqa: E402

PATH = "pkg/mod.py"
SOURCE = ("import os\n\n\ndef total(values):\n    acc = 0\n    for value in values:\n"
          "        acc += value\n    return acc\n\n\ndef unused():\n    return None\n")
LINES = SOURCE.splitlines(keepends=True)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / PATH).write_text(SOURCE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def handles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "handles")


def _runtime(root: Path) -> SimpleNamespace:
    return SimpleNamespace(owner=SimpleNamespace(root=str(root)),
                           _edit_protocol_state=EditProtocolRunState())


def _window(start: int, end: int) -> str:
    return f"[Working source: {PATH}]\n" + "".join(LINES[start - 1:end])


def _reader(path: str) -> str | None:
    return SOURCE if path == PATH else None


def _request(window: str) -> list[dict]:
    return [{"role": "system", "content": "You fix code."},
            {"role": "user", "content": "Task: make total() skip None."},
            {"role": "user", "content": window}]


def _frame_message(sites: list[dict]) -> dict:
    payload = {"state": {"framefit5": {"open_sites": sites}}}
    return {"role": "user", "content": "[StateFrame]\n" + json.dumps(payload, sort_keys=True)}


def _framed(frame: dict, *after: dict) -> list[dict]:
    """The request layout the engine sends: system, preface, the host frame, then the exchange."""
    return [{"role": "system", "content": "You fix code."},
            {"role": "user", "content": "Task: make total() skip None."}, frame, *after]


def _site(start: int, end: int, *, revision: str | None = None) -> dict:
    return {"path": PATH, "lines": {"start": start, "end": end, "total": len(LINES)},
            "revision": revision or source_revision(SOURCE),
            "current_bytes": "".join(LINES[start - 1:end]), "read_this_request": True}


def _maps_in(messages: list[dict]) -> list[int]:
    return [index for index, message in enumerate(messages)
            if CURRENT_SOURCE_PREFIX in str(message.get("content"))]


# --- OFF: legacy_safe is byte-identical ----------------------------------------------

def test_legacy_safe_returns_the_same_object(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    monkeypatch.delenv("STATEFRAME_EDIT_PROTOCOL", raising=False)
    messages = _request(_window(4, 8))
    before = json.dumps(messages)
    runtime = _runtime(workspace)
    assert request_map.apply(runtime, messages, request_ordinal=3) is messages
    assert json.dumps(messages) == before
    assert runtime._edit_protocol_state.handle_map is None


def test_explicit_legacy_safe_is_a_no_op(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "legacy_safe")
    messages = _request(_window(4, 8))
    assert request_map.apply(_runtime(workspace), messages, request_ordinal=3) is messages


def test_handles_without_a_bound_state_is_a_no_op(handles: None, workspace: Path) -> None:
    messages = _request(_window(4, 8))
    runtime = SimpleNamespace(owner=SimpleNamespace(root=str(workspace)))
    assert request_map.apply(runtime, messages, request_ordinal=3) is messages


# --- placement: map only in the final user message -----------------------------------

def test_map_lands_only_in_the_final_user_message(handles: None, workspace: Path) -> None:
    messages = _request(_window(4, 8))
    runtime = _runtime(workspace)
    placed = request_map.apply(runtime, messages, request_ordinal=3)
    assert _maps_in(placed) == [len(placed) - 1]
    assert placed[-1]["content"].startswith(messages[-1]["content"])
    assert all(placed[i] is messages[i] for i in range(len(messages) - 1))
    assert _maps_in(messages) == []                       # the caller's transcript stays map-free
    bound = runtime._edit_protocol_state.handle_map
    assert bound is not None and bound.request_ordinal == 3
    parsed = parse_map(placed[-1]["content"])
    assert parsed["files"][0]["handles"] == [
        {"handle": "q3h1", "start_line": 4, "end_line": 8, "kind": "SPAN"}]
    assert parsed["files"][0]["not_shown"] == [(1, 3), (9, len(LINES))]


def test_tool_reply_final_gets_a_new_user_message(handles: None, workspace: Path) -> None:
    messages = [*_request(_window(4, 8)),
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    placed = request_map.apply(_runtime(workspace), messages, request_ordinal=4)
    assert placed[-2] is messages[-1] and placed[-2]["content"] == "ok"
    assert placed[-1]["role"] == "user" and placed[-1]["content"].startswith(CURRENT_SOURCE_PREFIX)
    assert _maps_in(placed) == [len(placed) - 1]


def test_nothing_delivered_sends_no_map_and_clears_handles(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    request_map.apply(runtime, _request(_window(4, 8)), request_ordinal=1)
    messages = _request("Run the tests and report.")
    placed = request_map.apply(runtime, messages, request_ordinal=2)
    assert placed == messages and _maps_in(placed) == []
    assert runtime._edit_protocol_state.handle_map is None


# --- cache: N and N+1 share the retained prefix ---------------------------------------

def test_request_n_and_n_plus_1_share_the_retained_prefix(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    transcript = _request(_window(4, 8))
    wire_n = json.dumps(request_map.apply(runtime, transcript, request_ordinal=1))
    transcript = [*transcript, {"role": "assistant", "content": "Reading more."},
                  {"role": "user", "content": _window(1, 12)}]
    sent_n1 = request_map.apply(runtime, transcript, request_ordinal=2)
    wire_n1 = json.dumps(sent_n1)
    map_start = wire_n.index(json.dumps("\n\n" + CURRENT_SOURCE_PREFIX)[1:-1])
    shared = 0
    while shared < min(len(wire_n), len(wire_n1)) and wire_n[shared] == wire_n1[shared]:
        shared += 1
    assert shared >= map_start                         # everything before N's map is retained
    assert wire_n.count(CURRENT_SOURCE_PREFIX) == 1 and wire_n1.count(CURRENT_SOURCE_PREFIX) == 1
    assert _maps_in(sent_n1) == [len(sent_n1) - 1]
    handles_n1 = parse_map(sent_n1[-1]["content"])["files"][0]["handles"]
    assert handles_n1 == [{"handle": "q2h1", "start_line": 1, "end_line": len(LINES),
                           "kind": "REWRITE"}]


def test_minting_is_deterministic(handles: None, workspace: Path) -> None:
    messages = _request(_window(4, 8))
    first = request_map.apply(_runtime(workspace), messages, request_ordinal=7)
    second = request_map.apply(_runtime(workspace), messages, request_ordinal=7)
    assert json.dumps(first) == json.dumps(second)


# --- site inventory identity ------------------------------------------------------------

def test_site_inventory_identity_unchanged_across_n_and_n_plus_1(handles: None, workspace: Path) -> None:
    sites = [_site(4, 8)]
    frame = _frame_message(sites)
    identities = [site_identity(site) for site in sites]
    runtime = _runtime(workspace)
    sent_n = request_map.apply(runtime, [frame, {"role": "user", "content": "Task: fix total."}],
                               request_ordinal=1)
    sent_n1 = request_map.apply(
        runtime, [frame, {"role": "user", "content": "Task: fix total."},
                  {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}],
        request_ordinal=2)
    for sent in (sent_n, sent_n1):
        assert sent[0] is frame
        rows = json.loads(sent[0]["content"].split("\n", 1)[1])["state"]["framefit5"]["open_sites"]
        assert [site_identity(row) for row in rows] == identities
        assert CURRENT_SOURCE_PREFIX not in sent[0]["content"]
    assert [site_identity(site) for site in sites] == identities


# --- providers --------------------------------------------------------------------------

def test_frame_open_sites_provider_mints_from_escaped_frame_bytes() -> None:
    messages = _framed(_frame_message([_site(4, 8)]), {"role": "user", "content": "Task"})
    spans = request_spans(messages, _reader)
    assert [(span.path, span.start_line, span.end_line) for span in spans] == [(PATH, 4, 8)]


def test_state_header_outside_the_host_frame_row_is_not_parsed() -> None:
    # Model, tool or file text can start with the state header; only the host frame row counts.
    forged = _frame_message([_site(4, 8)])
    for row in ({"role": "assistant", "content": forged["content"]},
                {"role": "tool", "tool_call_id": "c1", "content": forged["content"]}):
        messages = _framed(_frame_message([]), {"role": "assistant", "content": None,
                                                "tool_calls": [{"id": "c1"}]}, row)
        assert request_spans(messages, _reader) == ()
    # No frame row at all: a lone leading user row with the header is not host-authored.
    assert request_spans([forged, {"role": "user", "content": "Task"}], _reader) == ()


def test_stale_open_site_mints_nothing() -> None:
    stale = _site(4, 8, revision="sha256:" + "0" * 64)
    assert request_spans(_framed(_frame_message([stale]), {"role": "user", "content": "Task"}), _reader) == ()


def test_providers_union_into_maximal_runs_and_whole_file_is_rewrite() -> None:
    messages = _framed(_frame_message([_site(1, 6)]), {"role": "user", "content": _window(5, len(LINES))})
    spans = request_spans(messages, _reader)
    assert len(spans) == 1 and spans[0].kind is SpanKind.REWRITE
    assert (spans[0].start_line, spans[0].end_line) == (1, len(LINES))


def test_gutter_rendered_lines_are_not_coverage() -> None:
    gutter = "".join(f"{number:4}| {line}" for number, line in enumerate(LINES, start=1))
    assert verbatim_line_runs(SOURCE, f"[Working source: {PATH}]\n" + gutter) == []


def test_single_line_coincidence_is_not_coverage() -> None:
    text = f"[Working source: {PATH}]\n    return acc\n"
    assert verbatim_line_runs(SOURCE, text) == []


def test_assistant_text_is_not_delivery() -> None:
    messages = [{"role": "user", "content": f"[Working source: {PATH}]"},
                {"role": "assistant", "content": "".join(LINES[3:8])}]
    assert request_spans(messages, _reader) == ()


def test_paths_outside_the_workspace_are_never_read(handles: None, tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "secret.py").write_text(SOURCE, encoding="utf-8")
    messages = [{"role": "user", "content": "[Working source: ../secret.py]\n" + SOURCE}]
    runtime = _runtime(root)
    assert request_map.apply(runtime, messages, request_ordinal=1) == messages
    assert runtime._edit_protocol_state.handle_map is None


# --- repair turn and history --------------------------------------------------------------

def test_repair_turn_keeps_its_own_map(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    request_map.apply(runtime, _request(_window(4, 8)), request_ordinal=1)
    bound = runtime._edit_protocol_state.handle_map
    repair = [*_request(_window(4, 8)), {"role": "assistant", "content": "edit"},
              {"role": "tool", "content": "refused"},
              {"role": "user", "content": "[EDIT REFUSED]\n" + bound.render()},
              {"role": "user", "content": "host note after the repair"}]
    assert current_turn_has_map(repair)
    assert request_map.apply(runtime, repair, request_ordinal=2) is repair
    assert runtime._edit_protocol_state.handle_map is bound


def test_map_in_an_older_turn_does_not_block_a_fresh_map(handles: None, workspace: Path) -> None:
    older, _ = compose_request_map(_request(_window(4, 8)), request_ordinal=1, read_source=_reader)
    messages = [*older, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": _window(1, 12)}]
    assert not current_turn_has_map(messages)
    placed, handle_map = compose_request_map(messages, request_ordinal=2, read_source=_reader)
    assert handle_map is not None and handle_map.request_ordinal == 2
    assert _maps_in(placed)[-1] == len(placed) - 1


# --- end to end with the executor: the first edit resolves --------------------------------

def test_first_edit_source_resolves_against_the_sent_map(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    request_map.apply(runtime, _request(_window(4, 8)), request_ordinal=1)
    state = runtime._edit_protocol_state
    args = {"edits": [{"handle": "q1h1", "old_text": "        acc += value\n",
                       "new_text": "        if value is not None:\n            acc += value\n"}]}
    outcome = execute_edit_source(workspace, args, state=state,
                                  read_bytes=lambda path: path.read_bytes(),
                                  write_bytes=lambda path, data: (path.write_bytes(data), (True, None))[1])
    assert outcome.status == APPLIED, outcome.observation()
    assert "if value is not None" in (workspace / PATH).read_text(encoding="utf-8")


def test_edit_outside_the_delivered_span_is_refused(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    request_map.apply(runtime, _request(_window(4, 8)), request_ordinal=1)
    args = {"edits": [{"handle": "q1h1", "old_text": "    return None\n", "new_text": "    return 0\n"}]}
    outcome = execute_edit_source(workspace, args, state=runtime._edit_protocol_state,
                                  read_bytes=lambda path: path.read_bytes(),
                                  write_bytes=lambda path, data: (path.write_bytes(data), (True, None))[1])
    assert outcome.status == REJECTED
    assert (workspace / PATH).read_text(encoding="utf-8") == SOURCE


# --- seam pins ----------------------------------------------------------------------------

def _engine(name: str) -> str:
    return (ENGINE / name).read_text(encoding="utf-8")


def test_transport_places_the_map_right_after_the_read_once_final_message() -> None:
    legacy = _engine("kernel_measurement.py")
    read_once = legacy.index("body[\"messages\"] = _read_once_final(wire_truth_runtime, body.get(\"messages\"))")
    placed = legacy.index("from kernel_edit_request_map import apply as _edit_request_map", read_once)
    assert placed - read_once < 400
    assert legacy.index("prepare_final_visibility(", placed) > placed
    assert legacy.count("kernel_edit_request_map") == 1
    qualified = _engine("kernel_final_wire_allocation.py")
    read_once = qualified.index('body["messages"] = append_final_message(context, body["messages"])')
    placed = qualified.index("from kernel_edit_request_map import apply as edit_request_map", read_once)
    assert placed - read_once < 400
    assert qualified.index("finalize_frame_candidate(", placed) > placed
    assert qualified.count("kernel_edit_request_map") == 1


def test_host_binds_the_run_state_only_under_handles() -> None:
    source = (ENGINE / "probes" / "react_repair.py").read_text(encoding="utf-8")
    bind = source.index("measurement._edit_protocol_state = _edit_protocol_state")
    assert "if _edit_protocol_handles():" in source[bind - 200:bind]
    assert source.count("measurement._edit_protocol_state = ") == 1


# --- switch pairs in the shipped context (EMPIRICAL_FIXES on) ------------------------------

@pytest.mark.parametrize("empirical", ["on", "off"])
@pytest.mark.parametrize("protocol", ["legacy_safe", "handles"])
def test_switch_pairs_with_the_shipped_empirical_profile(monkeypatch: pytest.MonkeyPatch, workspace: Path,
                                                         empirical: str, protocol: str) -> None:
    monkeypatch.setenv("STATEFRAME_EMPIRICAL_FIXES", empirical)
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", protocol)
    messages = _request(_window(4, 8))
    before = json.dumps(messages)
    placed = request_map.apply(_runtime(workspace), messages, request_ordinal=1)
    assert json.dumps(messages) == before
    if protocol == "legacy_safe":
        assert placed is messages
    else:
        assert _maps_in(placed) == [len(placed) - 1]


def test_invalid_protocol_value_fails_loud(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "handle")
    with pytest.raises(ValueError):
        request_map.apply(_runtime(workspace), _request(_window(4, 8)), request_ordinal=1)


# --- fail-open is journaled ---------------------------------------------------------------

class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, **fields: object) -> None:
        self.events.append((event, fields))


def test_mint_failure_sends_the_body_unchanged_and_journals(handles: None, workspace: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(workspace)
    request_map.apply(runtime, _request(_window(4, 8)), request_ordinal=1)

    def broken(*_args: object, **_kwargs: object) -> object:
        raise ValueError("planted")

    monkeypatch.setattr(request_map, "compose_request_map", broken)
    journal = _Journal()
    messages = _request(_window(4, 8))
    assert request_map.apply(runtime, messages, request_ordinal=2, journal=journal) is messages
    assert runtime._edit_protocol_state.handle_map is None
    assert journal.events == [("edit_handle_map_failed",
                               {"producer": "kernel_edit_request_map", "dispatch": False,
                                "request_ordinal": 2, "error_type": "ValueError"})]


def test_success_is_journaled_without_paths_or_bytes(handles: None, workspace: Path) -> None:
    journal = _Journal()
    request_map.apply(_runtime(workspace), _request(_window(4, 8)), request_ordinal=5, journal=journal)
    [(event, fields)] = journal.events
    assert event == "edit_handle_map" and fields["handles"] == 1 and fields["files"] == 1
    assert PATH not in json.dumps(fields) and "acc" not in json.dumps(fields)


@pytest.mark.parametrize("ordinal", [None, -1, "3", 2.0])
def test_bad_ordinal_clears_handles_and_sends_unchanged(handles: None, workspace: Path, ordinal: object) -> None:
    runtime = _runtime(workspace)
    request_map.apply(runtime, _request(_window(4, 8)), request_ordinal=1)
    messages = _request(_window(4, 8))
    assert request_map.apply(runtime, messages, request_ordinal=ordinal) is messages
    assert runtime._edit_protocol_state.handle_map is None


# --- boundary values ----------------------------------------------------------------------

def test_crlf_bom_and_non_ascii_bytes_are_addressed_exactly(handles: None, tmp_path: Path) -> None:
    text = "﻿import os\r\n\r\ndef naïve(x):\r\n    y = x  # café\r\n    return y\r\n"
    (tmp_path / "m.py").write_bytes(text.encode("utf-8"))
    runtime = _runtime(tmp_path)
    window = "[Working source: m.py]\n" + "".join(text.splitlines(keepends=True)[2:5])
    placed = request_map.apply(runtime, [{"role": "user", "content": window}], request_ordinal=1)
    record = runtime._edit_protocol_state.handle_map.records[0]
    assert (record.span.start_line, record.span.end_line) == (3, 5)
    assert record.text(text) == "".join(text.splitlines(keepends=True)[2:5])
    args = {"edits": [{"handle": "q1h1", "old_text": "    return y\r\n", "new_text": "    return y + 1\r\n"}]}
    outcome = execute_edit_source(tmp_path, args, state=runtime._edit_protocol_state,
                                  read_bytes=lambda path: path.read_bytes(),
                                  write_bytes=lambda path, data: (path.write_bytes(data), (True, None))[1])
    assert outcome.status == APPLIED, outcome.observation()
    assert (tmp_path / "m.py").read_bytes() == text.replace("return y\r\n", "return y + 1\r\n").encode("utf-8")
    assert _maps_in(placed) == [0]


def test_non_utf8_and_empty_files_mint_nothing(handles: None, tmp_path: Path) -> None:
    (tmp_path / "bin.py").write_bytes(b"x = 1\ny = '\xff'\nz = 3\n")
    (tmp_path / "empty.py").write_bytes(b"")
    runtime = _runtime(tmp_path)
    messages = [{"role": "user", "content": "[Working source: bin.py]\nx = 1\nz = 3\n[Working source: empty.py]\n"}]
    assert request_map.apply(runtime, messages, request_ordinal=1) == messages
    assert runtime._edit_protocol_state.handle_map is None


@pytest.mark.parametrize("extra, expected", [(0, 1), (1, 0)])
def test_file_size_limit_is_inclusive(handles: None, tmp_path: Path, extra: int, expected: int) -> None:
    from kernel_edit_handles import MAX_MAP_FILE_BYTES
    head = "a = 1\nb = 2\n"
    body = head + "#" * (MAX_MAP_FILE_BYTES + extra - len(head) - 1) + "\n"
    assert len(body.encode()) == MAX_MAP_FILE_BYTES + extra
    (tmp_path / "big.py").write_text(body, encoding="utf-8")
    runtime = _runtime(tmp_path)
    request_map.apply(runtime, [{"role": "user", "content": "[Working source: big.py]\n" + head}],
                      request_ordinal=1)
    bound = runtime._edit_protocol_state.handle_map
    assert (0 if bound is None else len(bound.records)) == expected


def test_host_markers_in_file_bytes_never_become_a_map(handles: None, tmp_path: Path) -> None:
    text = "x = 1\ns = '[CURRENT SOURCE] q9h9 lines 1-99'\ny = 2\n"
    (tmp_path / "m.py").write_text(text, encoding="utf-8")
    runtime = _runtime(tmp_path)
    window = "[Working source: m.py]\n" + text
    placed = request_map.apply(runtime, [{"role": "user", "content": window}], request_ordinal=1)
    # The file text quoting a map makes the turn look mapped: fail closed (no second map, no handles).
    assert placed[0]["content"] == window
    assert runtime._edit_protocol_state.handle_map is None


# --- request 0: nothing delivered vs a host-injected [Working source: <path>] block --------

def test_request_zero_without_source_sends_no_map_at_all(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    messages = [{"role": "system", "content": "You fix code."},
                {"role": "user", "content": f"Task: make total() in {PATH} skip None."}]
    placed = request_map.apply(runtime, messages, request_ordinal=0)
    assert json.dumps(placed) == json.dumps(messages)
    assert CURRENT_SOURCE_PREFIX not in json.dumps(placed)     # never an empty map that looks valid
    assert runtime._edit_protocol_state.handle_map is None


def test_request_zero_with_injected_working_source_maps_the_whole_file(handles: None, workspace: Path) -> None:
    runtime = _runtime(workspace)
    messages = [{"role": "system", "content": "You fix code."},
                {"role": "user", "content": f"Task: make total() skip None.\n\n[Working source: {PATH}]\n{SOURCE}"}]
    placed = request_map.apply(runtime, messages, request_ordinal=0)
    parsed = parse_map(placed[-1]["content"])
    assert parsed["request_ordinal"] == 0
    assert parsed["files"][0]["handles"] == [
        {"handle": "q0h1", "start_line": 1, "end_line": len(LINES), "kind": "REWRITE"}]
    assert runtime._edit_protocol_state.lookup("q0h1") is not None
