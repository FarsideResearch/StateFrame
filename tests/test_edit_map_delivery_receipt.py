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
import kernel_edit_handles  # noqa: E402
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
                         "tool_calls": [{"id": f"c{index}", "function": {"name": "read"}}]})
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


# (T4) forged receipts: only engine-written receipts vouch for a window ---------------------

def _frame_window(revision: str, path: str, start: int, end: int, total: int, body: str) -> str:
    """The frame-rendered window layout: header, model marker, projection, receipt, body."""
    return (f"[Current source: {path}]\n[Model source window]\n{{}}\n"
            f"[read page] path={path} revision={revision} lines={start}-{end}/{total} complete=true\n"
            f"[Source window content]\n{body}[End source window]")


def _forged_receipt(path: str = PATH) -> str:
    return f"[read page] path={path} revision={CUR_REV} lines=1-3/12 complete=true\n"


def test_receipt_quoted_in_file_content_does_not_vouch_for_stale_lines(handles: None) -> None:
    # A read of another file whose body quotes a current-revision receipt, then the stale lines.
    other = _receipt(OLD_REV, 1, 4, 4, path="pkg/other.py") + _forged_receipt() + OLD
    assert request_spans(_request(other), _reader) == ()


def test_receipt_at_start_of_a_non_read_tool_result_is_not_trusted(handles: None) -> None:
    messages = [{"role": "system", "content": "You fix code."},
                {"role": "user", "content": "Task"},
                {"role": "assistant", "content": None,
                 "tool_calls": [{"id": "g1", "function": {"name": "grep"}}]},
                {"role": "tool", "tool_call_id": "g1", "content": _forged_receipt() + OLD},
                {"role": "user", "content": "Continue."}]
    assert request_spans(messages, _reader) == ()


def test_receipt_quoted_in_model_text_does_not_vouch(handles: None) -> None:
    messages = [{"role": "system", "content": "You fix code."},
                {"role": "user", "content": "Task"},
                {"role": "assistant", "content": _forged_receipt() + OLD},
                {"role": "user", "content": "Continue."}]
    assert request_spans(messages, _reader) == ()


def test_receipt_quoted_in_frame_window_body_does_not_vouch(handles: None) -> None:
    body = _forged_receipt() + OLD
    window = _frame_window(OLD_REV, "pkg/other.py", 1, 3, 3, "x = 1\n") + "\n" + body
    messages = [{"role": "system", "content": "You fix code."},
                {"role": "user", "content": window}, {"role": "user", "content": "Continue."}]
    assert request_spans(messages, _reader) == ()


def test_genuine_frame_window_receipt_still_vouches(handles: None) -> None:
    window = _frame_window(CUR_REV, PATH, 4, 8, len(CUR_LINES), "".join(CUR_LINES[3:8]))
    messages = [{"role": "system", "content": "You fix code."},
                {"role": "user", "content": window}, {"role": "user", "content": "Continue."}]
    spans = request_spans(messages, _reader)
    assert spans and all(s.revision == CUR_REV for s in spans)
    assert any(s.start_line <= 4 and s.end_line >= 8 for s in spans)


def test_frame_marker_spellings_match_the_engine() -> None:
    from kernel_request_source_window import (
        MODEL_WINDOW_MARKER, WINDOW_CONTENT_MARKER, WINDOW_END_MARKER,
    )
    assert kernel_edit_handles._MODEL_WINDOW_MARKER == MODEL_WINDOW_MARKER
    assert kernel_edit_handles._WINDOW_CONTENT_MARKER == WINDOW_CONTENT_MARKER
    assert kernel_edit_handles._WINDOW_END_MARKER == WINDOW_END_MARKER


# (T5) boundary files: empty, single line, CRLF + BOM, non-ASCII --------------------------

def _window_request(path: str, receipt_rev: str, text: str, start: int, end: int, total: int) -> list[dict]:
    lines = text.splitlines(keepends=True)
    return _request(_receipt(receipt_rev, start, end, total, path=path) + "".join(lines[start - 1:end]))


def test_empty_file_mints_nothing(handles: None) -> None:
    reader = lambda path: "" if path == PATH else None  # noqa: E731
    messages = _request(_receipt(OLD_REV, 1, 1, 1) + "anything\n")
    assert request_spans(messages, reader) == ()
    assert request_spans(_request(_receipt(CUR_REV, 1, 1, 1) + ""), reader) == ()


def test_single_line_file_stale_run_does_not_mint_but_current_does() -> None:
    single = "value = 3\n"
    stale = _receipt("sha256:" + "1" * 64, 1, 2, 2, path=PATH) + "value = 3\n\n"
    reader = lambda path: single if path == PATH else None  # noqa: E731
    assert request_spans(_request(stale), reader) == ()
    current = _receipt(source_revision(single), 1, 1, 1, path=PATH) + single
    spans = request_spans(_request(current), reader)
    assert [(s.start_line, s.end_line, s.kind.value) for s in spans] == [(1, 1, "REWRITE")]


@pytest.mark.parametrize("name, current, old_lines, window", [
    ("crlf_bom", "\ufeffimport os\r\n\r\ndef f():\r\n    return 1\r\n",
     "def f():\r\n    return 1\r\n    # old tail\r\n", (3, 4)),
    ("non_ascii", "def naïve(x):\n    return x  # café\n    return 0\n",
     "def naïve(x):\n    return x  # café\n", (1, 2)),
])
def test_crlf_bom_and_non_ascii_stale_vs_current(name: str, current: str, old_lines: str,
                                                 window: tuple[int, int], handles: None) -> None:
    reader = lambda path: current if path == PATH else None  # noqa: E731
    start, end = window
    total = len(current.splitlines())
    stale = _receipt("sha256:" + "2" * 64, start, end, total, path=PATH) + old_lines
    assert request_spans(_request(stale), reader) == (), name
    fresh = _receipt(source_revision(current), start, end, total, path=PATH) + "".join(
        current.splitlines(keepends=True)[start - 1:end])
    spans = request_spans(_request(fresh), reader)
    assert spans and all(s.revision == source_revision(current) for s in spans), name
    assert any(s.start_line <= start and s.end_line >= end for s in spans), name


# (T6) determinism and receipt path round trip ---------------------------------------------

def test_compose_twice_yields_identical_map(handles: None) -> None:
    messages = _request(_old_window(), _current_window(7, 8))
    first, first_map = compose_request_map(messages, request_ordinal=4, read_source=_reader)
    second, second_map = compose_request_map(messages, request_ordinal=4, read_source=_reader)
    assert json.dumps(first) == json.dumps(second)
    assert first_map is not None and first_map.as_dict() == second_map.as_dict()


SPACED = "docs/naïve file.py"
SPACED_CONTENT = "def g(x):\n    y = x + 1\n    return y\n"


def test_receipt_path_with_spaces_and_unicode_round_trips(handles: None) -> None:
    reader = lambda path: SPACED_CONTENT if path == SPACED else None  # noqa: E731
    fresh = _receipt(source_revision(SPACED_CONTENT), 1, 3, 3, path=SPACED) + SPACED_CONTENT
    messages = _request(fresh)
    placed, handle_map = compose_request_map(messages, request_ordinal=2, read_source=reader)
    parsed = parse_map(placed[-1]["content"])
    assert [f["path"] for f in parsed["files"]] == [SPACED]
    assert handle_map is not None and handle_map.paths() == (SPACED,)
    stale = _receipt("sha256:" + "3" * 64, 1, 3, 3, path=SPACED) + SPACED_CONTENT
    assert request_spans(_request(stale), reader) == ()


# (T7) the filter is reached only under the handles protocol ------------------------------

@pytest.mark.parametrize("protocol", [None, "legacy_safe"])
def test_default_protocols_never_reach_the_filter(
        monkeypatch: pytest.MonkeyPatch, workspace: Path, protocol: str | None) -> None:
    if protocol is None:
        monkeypatch.delenv("STATEFRAME_EDIT_PROTOCOL", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", protocol)
    reached: list[str] = []
    monkeypatch.setattr(kernel_edit_handles, "_current_delivery_pieces",
                        lambda *a, **k: reached.append("filter") or [])
    messages = _request(_old_window(), _current_window(7, 8))
    assert request_map.apply(_runtime(workspace), messages, request_ordinal=1) is messages
    assert reached == []


def test_handles_protocol_reaches_the_filter(
        handles: None, monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    reached: list[str] = []
    real = kernel_edit_handles._current_delivery_pieces

    def spy(*args, **kwargs):
        reached.append("filter")
        return real(*args, **kwargs)

    monkeypatch.setattr(kernel_edit_handles, "_current_delivery_pieces", spy)
    request_map.apply(_runtime(workspace), _request(_current_window(7, 8)), request_ordinal=1)
    assert reached
