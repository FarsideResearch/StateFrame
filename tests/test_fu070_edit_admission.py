"""FU-070 request-bound edit admission: handle issuance, admission reasons, and path expiry."""
from __future__ import annotations

import hashlib
import io

import pytest

from kernel_edit_admission import EditHandle, EditNotAdmitted, admit_edit, expire_handles, issue_handles
from kernel_source_view import SourceView, VerifiedSpan


def _sha(data: bytes) -> str:
    # Independent of the module under test so a broken _sha cannot hide itself.
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _span(source: bytes, start: int, end: int) -> VerifiedSpan:
    lines = io.BytesIO(source).readlines()
    return VerifiedSpan(start, end, _sha(b"".join(lines[start - 1:end])))


def _view(source: bytes, *, spans: tuple[VerifiedSpan, ...], complete: bool = True,
          path: str = "owned.py") -> SourceView:
    return SourceView(path=path, source_sha256=_sha(source),
                      total_lines=len(io.BytesIO(source).readlines()),
                      eof_newline=source.endswith(b"\n"), spans=spans, complete=complete)


SIX = b"l1\nl2\nl3\nl4\nl5\nl6\n"


def _file_handle(source: bytes, spans: tuple[VerifiedSpan, ...]) -> EditHandle:
    return EditHandle("req", "owned.py", _sha(source), spans, "FILE")


def _span_handle(source: bytes, spans: tuple[VerifiedSpan, ...]) -> EditHandle:
    return EditHandle("req", "owned.py", _sha(source), spans, "SPAN")


def _reason(handle: EditHandle, source: bytes, touched: tuple[int, int]) -> str:
    with pytest.raises(EditNotAdmitted) as raised:
        admit_edit(handle, current_source=source, touched_lines=touched)
    assert raised.value.path == "owned.py"
    return raised.value.reason


# issue_handles

def test_incomplete_view_issues_no_handle() -> None:
    view = _view(SIX, spans=(_span(SIX, 1, 3),), complete=False)
    assert issue_handles(view, request_id="req-1") is None


def test_complete_view_issues_file_handle_bound_to_request_and_revision() -> None:
    spans = (_span(SIX, 1, 6),)
    handle = issue_handles(_view(SIX, spans=spans), request_id="req-1")
    assert handle == EditHandle("req-1", "owned.py", _sha(SIX), spans, "FILE")
    assert handle is not None and handle.mode == "FILE"


@pytest.mark.parametrize("request_id", ["", "   ", "\t\n"])
def test_blank_request_id_is_rejected(request_id: str) -> None:
    with pytest.raises(ValueError):
        issue_handles(_view(SIX, spans=(_span(SIX, 1, 6),)), request_id=request_id)


def test_blank_request_id_rejected_even_for_incomplete_view() -> None:
    with pytest.raises(ValueError):
        issue_handles(_view(SIX, spans=(), complete=False), request_id="")


def test_issued_file_handle_admits_edit_anywhere_in_file() -> None:
    handle = issue_handles(_view(SIX, spans=(_span(SIX, 1, 6),)), request_id="req-1")
    assert handle is not None
    admit_edit(handle, current_source=SIX, touched_lines=(2, 5))


# admit_edit: happy paths

def test_file_mode_happy_path_admits_touched_range_across_split_spans() -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 2), _span(SIX, 3, 6)))
    admit_edit(handle, current_source=SIX, touched_lines=(2, 5))


def test_span_mode_happy_path_admits_touched_range_inside_span() -> None:
    handle = _span_handle(SIX, (_span(SIX, 2, 3),))
    admit_edit(handle, current_source=SIX, touched_lines=(2, 3))


# admit_edit: one reason code per case

def test_stale_revision_when_current_source_changed() -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 6),))
    changed = SIX + b"l7\n"
    assert _reason(handle, changed, (1, 1)) == "stale_revision"


@pytest.mark.parametrize("touched", [(0, 1), (2, 1), (1, 7), (7, 7), [1, 2], (1, 2, 3), (True, 2), (1.0, 2)])
def test_invalid_range_for_bad_bounds(touched: object) -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 6),))
    assert _reason(handle, SIX, touched) == "invalid_range"  # type: ignore[arg-type]


def test_span_bytes_mismatch_when_one_byte_changes_inside_span() -> None:
    # Handle claims the mutated revision, but its span bytes are from the original.
    mutated = b"l1\nX2\nl3\nl4\nl5\nl6\n"
    handle = _file_handle(mutated, (_span(SIX, 1, 6),))
    assert _reason(handle, mutated, (1, 1)) == "span_bytes_mismatch"


def test_span_out_of_file_bounds_is_span_bytes_mismatch() -> None:
    handle = _span_handle(SIX, (VerifiedSpan(5, 9, _sha(b"l5\nl6\n")),))
    assert _reason(handle, SIX, (5, 6)) == "span_bytes_mismatch"


def test_file_incomplete_when_tail_is_uncovered() -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 3),))
    assert _reason(handle, SIX, (1, 1)) == "file_incomplete"


def test_overlapping_spans_cannot_fill_a_gap() -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 3), _span(SIX, 2, 4)))
    assert _reason(handle, SIX, (1, 1)) == "file_incomplete"


def test_file_incomplete_with_gap_in_middle() -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 2), _span(SIX, 4, 6)))
    assert _reason(handle, SIX, (1, 1)) == "file_incomplete"


def test_span_mode_outside_span_is_refused() -> None:
    handle = _span_handle(SIX, (_span(SIX, 1, 3),))
    assert _reason(handle, SIX, (4, 4)) == "outside_span"


def test_range_straddling_span_boundary_is_outside_span() -> None:
    handle = _span_handle(SIX, (_span(SIX, 1, 3),))
    assert _reason(handle, SIX, (3, 4)) == "outside_span"


def test_range_straddling_two_adjacent_spans_is_admitted() -> None:
    handle = _span_handle(SIX, (_span(SIX, 1, 3), _span(SIX, 4, 6)))
    admit_edit(handle, current_source=SIX, touched_lines=(3, 4))


# admit_edit: ordering

def test_stale_revision_wins_over_invalid_range_and_span_mismatch() -> None:
    handle = _file_handle(SIX, (VerifiedSpan(1, 9, _sha(b"x")),))
    assert _reason(handle, SIX + b"extra\n", (99, 99)) == "stale_revision"


def test_invalid_range_wins_over_span_mismatch() -> None:
    handle = _file_handle(SIX, (VerifiedSpan(1, 6, _sha(b"x")),))
    assert _reason(handle, SIX, (9, 9)) == "invalid_range"


def test_span_mismatch_wins_over_file_incomplete() -> None:
    handle = _file_handle(SIX, (VerifiedSpan(1, 3, _sha(b"x")),))
    assert _reason(handle, SIX, (1, 1)) == "span_bytes_mismatch"


def test_file_incomplete_wins_over_outside_span() -> None:
    handle = _file_handle(SIX, (_span(SIX, 1, 3),))
    assert _reason(handle, SIX, (6, 6)) == "file_incomplete"


# admit_edit: line counting

def test_unterminated_last_line_counts_as_a_line() -> None:
    source = b"a\nb\nc"  # 3 lines, no trailing newline
    handle = _file_handle(source, (_span(source, 1, 3),))
    admit_edit(handle, current_source=source, touched_lines=(3, 3))
    assert _reason(handle, source, (4, 4)) == "invalid_range"


def test_terminated_file_does_not_gain_a_phantom_line() -> None:
    source = b"a\nb\nc\n"  # 3 lines
    handle = _file_handle(source, (_span(source, 1, 3),))
    assert _reason(handle, source, (4, 4)) == "invalid_range"


def test_empty_source_has_no_admissible_lines() -> None:
    handle = _file_handle(b"", ())
    assert _reason(handle, b"", (1, 1)) == "invalid_range"


# expire_handles

def _handles() -> tuple[EditHandle, ...]:
    return (
        EditHandle("r", "a.py", _sha(b"a"), (), "FILE"),
        EditHandle("r", "b.py", _sha(b"b"), (), "FILE"),
        EditHandle("r", "sub/c.py", _sha(b"c"), (), "SPAN"),
    )


def test_expire_removes_touched_path_and_keeps_others_in_order() -> None:
    remaining = expire_handles(_handles(), touched_paths=["a.py"])
    assert [h.path for h in remaining] == ["b.py", "sub/c.py"]
    assert isinstance(remaining, tuple)


def test_expire_normalizes_touched_and_handle_paths() -> None:
    remaining = expire_handles(_handles(), touched_paths=["./sub//c.py"])
    assert [h.path for h in remaining] == ["a.py", "b.py"]


def test_expire_with_no_touched_paths_keeps_everything() -> None:
    assert expire_handles(_handles(), touched_paths=[]) == _handles()


def test_expire_allows_dots_inside_a_file_name() -> None:
    handles = (EditHandle("r", "data..bak.py", _sha(b"d"), (), "FILE"),)
    assert expire_handles(handles, touched_paths=["data..bak.py"]) == ()


@pytest.mark.parametrize("path", ["../x.py", "a/../b.py", "..", "/etc/passwd", "/a.py", ""])
def test_expire_rejects_traversal_absolute_and_empty_paths(path: str) -> None:
    with pytest.raises(ValueError):
        expire_handles(_handles(), touched_paths=[path])


# admit_unit_edit: the subframe edit gate over a frozen request's messages

from kernel_edit_admission import admit_unit_edit  # noqa: E402
from kernel_source_view import render_source_map  # noqa: E402
from kernel_working_set_windows import render_frozen_span  # noqa: E402

UNIT_PATH = "pkg/big.py"
UNIT_ROWS = [f"VALUE_{n:04d} = {n}\n" for n in range(1, 31)]
UNIT_SOURCE = "".join(UNIT_ROWS).encode()
UNIT_SHA = _sha(UNIT_SOURCE)


def _unit_page(start: int, end: int, *, source: bytes = UNIT_SOURCE,
               revision: str = UNIT_SHA) -> dict[str, str]:
    rows = source.decode().splitlines(keepends=True)
    page, _ = render_frozen_span(
        {"path": UNIT_PATH, "revision": revision, "total_lines": len(rows)},
        {"start_line": start, "end_line": end}, "".join(rows[start - 1:end]))
    return {"role": "user", "content": page}


def _unit_map(*spans: tuple[int, int], revision: str = UNIT_SHA) -> dict[str, str]:
    return {"role": "user", "content": render_source_map({UNIT_PATH: (revision, spans)})}


def _never_legacy(content: object) -> bool:
    raise AssertionError("legacy check must not run for this source")


def _no_legacy(content: object) -> bool:
    return False


def test_unit_edit_complete_map_and_pages_is_admitted() -> None:
    messages = [_unit_page(1, 10), _unit_page(11, 20), _unit_page(21, 30),
                _unit_map((1, 10), (11, 20), (21, 30))]
    assert admit_unit_edit(messages, path=UNIT_PATH, source=UNIT_SOURCE, request_id="req-1",
                           legacy_complete=_no_legacy) == (True, "", "")


def test_unit_edit_missing_middle_page_is_refused_with_missing_span() -> None:
    messages = [_unit_page(1, 10), _unit_page(21, 30), _unit_map((1, 10), (21, 30))]
    ok, reason, detail = admit_unit_edit(messages, path=UNIT_PATH, source=UNIT_SOURCE,
                                         request_id="req-1", legacy_complete=_no_legacy)
    assert ok is False
    assert reason == "incomplete_view"
    assert detail == ("edit needs the complete current source of pkg/big.py; "
                      "read the missing lines first: 11-20")
    assert len(detail) <= 300


def test_unit_edit_no_pages_refuses_with_whole_file_range() -> None:
    ok, reason, detail = admit_unit_edit([], path=UNIT_PATH, source=UNIT_SOURCE,
                                         request_id="req-1", legacy_complete=_no_legacy)
    assert (ok, reason) == (False, "incomplete_view")
    assert detail.endswith("read the missing lines first: 1-30")


def test_unit_edit_legacy_full_block_is_admitted_without_pages() -> None:
    seen: list[object] = []

    def legacy(content: object) -> bool:
        seen.append(content)
        return content == "legacy-block"

    messages = [{"role": "user", "content": "legacy-block"}]
    assert admit_unit_edit(messages, path=UNIT_PATH, source=UNIT_SOURCE, request_id="req-1",
                           legacy_complete=legacy) == (True, "", "")
    assert seen == ["legacy-block"]


def test_unit_edit_stale_page_from_old_revision_is_refused() -> None:
    old_source = UNIT_SOURCE.replace(b"VALUE_0005 = 5", b"VALUE_0005 = 55")
    old_sha = _sha(old_source)
    messages = [_unit_page(1, 10, source=old_source, revision=old_sha),
                _unit_page(11, 20), _unit_page(21, 30),
                _unit_map((11, 20), (21, 30))]
    ok, reason, detail = admit_unit_edit(messages, path=UNIT_PATH, source=UNIT_SOURCE,
                                         request_id="req-1", legacy_complete=_no_legacy)
    assert (ok, reason) == (False, "incomplete_view")
    assert detail.endswith("read the missing lines first: 1-10")


def test_unit_edit_empty_source_is_admitted() -> None:
    assert admit_unit_edit([], path="empty.py", source=b"", request_id="req-1",
                           legacy_complete=_never_legacy) == (True, "", "")


def test_unit_edit_non_utf8_source_skips_legacy_and_refuses() -> None:
    binary = b"\xff\xfe\x00 not text\n"
    ok, reason, detail = admit_unit_edit([{"role": "user", "content": "anything"}],
                                         path="blob.py", source=binary, request_id="req-1",
                                         legacy_complete=_never_legacy)
    assert (ok, reason) == (False, "incomplete_view")
    assert detail.startswith("edit needs the complete current source of blob.py")


def test_unit_edit_blank_request_id_is_refused_not_raised() -> None:
    ok, reason, _ = admit_unit_edit([], path=UNIT_PATH, source=UNIT_SOURCE, request_id=" ",
                                    legacy_complete=_no_legacy)
    assert (ok, reason) == (False, "no_request_id")


def test_unit_edit_refusal_detail_is_capped_at_300_characters() -> None:
    # One-line pages on every even line leave ~800 gaps, forcing the 300-char cut.
    rows = [f"V{n:05d} = {n}\n" for n in range(1, 1601)]
    source = "".join(rows).encode()
    pages = []
    for start in range(1, 1601, 2):
        page, _ = render_frozen_span(
            {"path": UNIT_PATH, "revision": _sha(source), "total_lines": 1600},
            {"start_line": start, "end_line": start},
            rows[start - 1])
        pages.append({"role": "user", "content": page})
    ok, reason, detail = admit_unit_edit(pages, path=UNIT_PATH, source=source, request_id="req-1",
                                         legacy_complete=_no_legacy)
    assert (ok, reason) == (False, "incomplete_view")
    assert len(detail) <= 300 and detail.endswith("...")
