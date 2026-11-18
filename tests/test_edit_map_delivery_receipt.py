"""Edit handle map: a source span is minted only from a window delivered at the current revision.

The map's text provider (kernel_edit_handles.request_spans, provider B) finds runs of current
file lines that the request text carries verbatim. Without a revision check, text delivered
under an OLD revision that happens to share a 2+ line run with the current file mints a handle
at the current revision. Read receipts (``[read page] path=P revision=sha256:H ...``) name the
bytes a window was delivered from, so a receipt at another revision (or for another path) masks
its text for this file. Receipt-less text keeps the legacy behavior.

Drives the production send seam (kernel_edit_request_map.apply) and the minter it calls.
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
    CURRENT_SOURCE_PREFIX, compose_request_map, parse_map, request_spans,
)
from kernel_edit_source_tool import EditProtocolRunState  # noqa: E402
from kernel_paged_source import source_revision  # noqa: E402

PATH = "pkg/mod.py"
CURRENT = ("import os\n\n\ndef total(values):\n    acc = 0\n    for value in values:\n"
           "        acc += value\n    return acc\n\n\ndef unused():\n    return None\n")
# The file as it was before a revert: its first two lines equal current lines 4-5 exactly.
OLD = "def total(values):\n    acc = 0\n    return acc\n"
CUR_LINES = CURRENT.splitlines(keepends=True)
CUR_REV = source_revision(CURRENT)
OLD_REV = source_revision(OLD)


def _runtime(root: Path) -> SimpleNamespace:
    return SimpleNamespace(owner=SimpleNamespace(root=str(root)),
                           _edit_protocol_state=EditProtocolRunState())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / PATH).write_text(CURRENT, encoding="utf-8")
    return tmp_path


@pytest.fixture
def handles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", "handles")


def _receipt(revision: str, start: int, end: int, total: int, path: str = PATH) -> str:
    return (f"[read page] path={path} revision={revision} "
            f"lines={start}-{end}/{total} complete=true\n")


def _current_window(start: int, end: int) -> str:
    return _receipt(CUR_REV, start, end, len(CUR_LINES)) + "".join(CUR_LINES[start - 1:end])


def _old_window() -> str:
    return _receipt(OLD_REV, 1, 3, 3) + OLD


def _request(*tool_reads: str) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "You fix code."},
                            {"role": "user", "content": "Task: make total() skip None."}]
    for index, read in enumerate(tool_reads):
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": f"c{index}"}]})
        messages.append({"role": "tool", "tool_call_id": f"c{index}", "content": read})
    messages.append({"role": "user", "content": "Continue."})
    return messages


def _plain_window(start: int, end: int) -> str:
    """The same lines under the legacy header label, with no read receipt (baseline only)."""
    return f"[Working source: {PATH}]\n" + "".join(CUR_LINES[start - 1:end])


def _reader(path: str) -> str | None:
    return CURRENT if path == PATH else None


# (a) stale revision text must not mint at the current revision ---------------------------

def test_old_revision_run_that_matches_current_lines_mints_nothing(
        handles: None, workspace: Path) -> None:
    # Precondition: the old window really does share a 2+ line verbatim run with the current file.
    assert "".join(CUR_LINES[3:5]) in OLD
    messages = _request(_old_window())
    assert request_spans(messages, _reader) == ()

    runtime = _runtime(workspace)
    # Nothing delivered at the current revision: the body goes out with the same messages, no map.
    assert request_map.apply(runtime, messages, request_ordinal=1) == messages
    assert runtime._edit_protocol_state.handle_map is None


def test_old_revision_run_mints_nothing_through_compose(handles: None) -> None:
    messages = _request(_old_window())
    placed, handle_map = compose_request_map(messages, request_ordinal=1, read_source=_reader)
    assert handle_map is None and placed == messages


# (b) a window delivered at the current revision mints exactly as before ------------------

def test_current_revision_window_mints_the_same_span(handles: None, workspace: Path) -> None:
    # The baseline is the same window without its receipt: the legacy text provider's answer.
    # The joined delivery text carries a blank line after the window, so the run may end one
    # line past the window; the receipt must not change that.
    baseline_placed, _ = compose_request_map(_request(_plain_window(4, 8)), request_ordinal=3,
                                             read_source=_reader)
    baseline = parse_map(baseline_placed[-1]["content"])["files"][0]
    runtime = _runtime(workspace)
    placed = request_map.apply(runtime, _request(_current_window(4, 8)), request_ordinal=3)
    parsed = parse_map(placed[-1]["content"])
    assert parsed["request_ordinal"] == 3
    assert parsed["files"] == [baseline]
    assert baseline["revision"] == CUR_REV and baseline["handles"][0]["start_line"] == 4
    assert baseline["handles"][0]["handle"] == "q3h1"
    assert runtime._edit_protocol_state.handle_map.records[0].span.revision == CUR_REV


def test_whole_file_at_current_revision_is_a_rewrite(handles: None) -> None:
    whole = _current_window(1, len(CUR_LINES))
    spans = request_spans(_request(whole), _reader)
    assert [(s.start_line, s.end_line, s.kind.value) for s in spans] == [
        (1, len(CUR_LINES), "REWRITE")]


# Mixed: stale and current windows of the same file in one request ------------------------

def test_only_the_current_window_of_a_mixed_request_is_minted(handles: None) -> None:
    # The model read the old revision earlier (its text matches current lines 4-5) and the
    # current revision later (lines 7-8). Legacy text matching also mints 4-5 at CUR_REV.
    messages = _request(_old_window(), _current_window(7, 8))
    spans = request_spans(messages, _reader)
    only_current = request_spans(_request(_plain_window(7, 8)), _reader)
    assert [(s.revision, s.start_line, s.end_line) for s in spans] == [
        (s.revision, s.start_line, s.end_line) for s in only_current]
    assert spans and all(s.revision == CUR_REV and s.start_line >= 7 for s in spans)


# (c) default protocol: request bytes never change --------------------------------------

@pytest.mark.parametrize("protocol", [None, "legacy_safe"])
def test_default_protocol_request_bytes_are_unchanged(
        monkeypatch: pytest.MonkeyPatch, workspace: Path, protocol: str | None) -> None:
    if protocol is None:
        monkeypatch.delenv("STATEFRAME_EDIT_PROTOCOL", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", protocol)
    messages = _request(_old_window(), _current_window(7, 8))
    before = json.dumps(messages, sort_keys=True)
    runtime = _runtime(workspace)
    assert request_map.apply(runtime, messages, request_ordinal=2) is messages
    assert json.dumps(messages, sort_keys=True) == before
    assert runtime._edit_protocol_state.handle_map is None
