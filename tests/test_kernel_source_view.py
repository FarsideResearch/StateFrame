"""FU-070-ALLOC: byte-verified source view, current-source map, deferred-first gate."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from kernel_source_view import (
    SOURCE_MAP_HEADER, SourceIncomplete, deferred_read_obligation_valid, file_floor_messages,
    parse_source_map, render_source_map, require_complete_source, source_lines, source_view,
)
from kernel_working_set_windows import render_frozen_span

PATH = "pkg/big.py"
LINES = [f"VALUE_{n:04d} = {n}\n" for n in range(1, 31)]
SOURCE = "".join(LINES).encode()
SHA = "sha256:" + hashlib.sha256(SOURCE).hexdigest()


def _window(start: int, end: int, *, source: bytes = SOURCE, path: str = PATH,
            revision: str | None = None, lines: list[str] | None = None) -> dict[str, str]:
    rows = lines if lines is not None else source.decode().splitlines(keepends=True)
    page, _ = render_frozen_span(
        {"path": path, "revision": revision or "sha256:" + hashlib.sha256(source).hexdigest(),
         "total_lines": len(rows)},
        {"start_line": start, "end_line": end}, "".join(rows[start - 1:end]))
    return {"role": "user", "content": page}


def _map(*spans: tuple[int, int], revision: str = SHA, path: str = PATH) -> dict[str, str]:
    return {"role": "user", "content": render_source_map({path: (revision, spans)})}


def _three_pages() -> list[dict[str, str]]:
    return [_window(1, 10), _window(11, 20), _window(21, 30)]


def test_full_raw_block_is_complete_without_a_map() -> None:
    messages = [{"role": "user", "content": f"[Current source: {PATH}]\n{SOURCE.decode()}"}]
    view = source_view(messages, path=PATH, source=SOURCE)
    assert view.complete and view.total_lines == 30 and view.eof_newline
    assert view.source_sha256 == SHA
    assert [(s.start, s.end) for s in view.spans] == [(1, 30)]
    assert view.spans[0].sha256 == SHA


def test_three_pages_and_map_are_complete() -> None:
    view = source_view([*_three_pages(), _map((1, 10), (11, 20), (21, 30))],
                       path=PATH, source=SOURCE)
    assert view.complete
    assert [(s.start, s.end) for s in view.spans] == [(1, 30)]  # merged only after verification


def test_pages_without_a_map_are_not_complete() -> None:
    assert not source_view(_three_pages(), path=PATH, source=SOURCE).complete


def test_middle_page_removed_is_incomplete_with_missing_span() -> None:
    messages = [_window(1, 10), _window(21, 30), _map((1, 10), (21, 30))]
    view = source_view(messages, path=PATH, source=SOURCE)
    assert not view.complete
    assert [(s.start, s.end) for s in view.spans] == [(1, 10), (21, 30)]
    events: list[tuple[str, dict[str, object]]] = []
    journal = SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields)))
    with pytest.raises(SourceIncomplete) as caught:
        require_complete_source(messages, path=PATH, source=SOURCE, journal=journal)
    assert caught.value.missing_spans == ((11, 20),)
    assert caught.value.path == PATH
    assert events[0][0] == "source_view_incomplete"
    assert events[0][1]["missing_spans"] == [[11, 20]] and events[0][1]["source_sha256"] == SHA


def test_require_complete_returns_the_view_and_tolerates_a_journal_without_emit() -> None:
    view = require_complete_source([*_three_pages(), _map((1, 30))], path=PATH, source=SOURCE,
                                   journal=None)
    assert view.complete
    with pytest.raises(SourceIncomplete):
        require_complete_source([], path=PATH, source=SOURCE, journal=object())


def test_edited_page_text_is_not_counted() -> None:
    forged = _window(11, 20)
    forged["content"] = forged["content"].replace("VALUE_0015 = 15", "VALUE_0015 = 99")
    messages = [_window(1, 10), forged, _window(21, 30), _map((1, 30))]
    view = source_view(messages, path=PATH, source=SOURCE)
    assert not view.complete
    assert [(s.start, s.end) for s in view.spans] == [(1, 10), (21, 30)]
    with pytest.raises(SourceIncomplete) as caught:
        require_complete_source(messages, path=PATH, source=SOURCE, journal=None)
    assert caught.value.missing_spans == ((11, 20),)


def test_stale_revision_window_is_not_counted() -> None:
    stale_source = SOURCE.replace(b"VALUE_0015 = 15", b"VALUE_0015 = 16")
    stale = _window(11, 20, source=stale_source)
    view = source_view([_window(1, 10), stale, _window(21, 30), _map((1, 30))],
                       path=PATH, source=SOURCE)
    assert not view.complete
    assert [(s.start, s.end) for s in view.spans] == [(1, 10), (21, 30)]


def test_window_for_another_path_is_not_counted() -> None:
    other = _window(1, 30, path="pkg/other.py")
    assert source_view([other, _map((1, 30))], path=PATH, source=SOURCE).spans == ()


def test_overlap_never_fills_a_hole() -> None:
    messages = [_window(1, 10), _window(5, 15), _window(21, 30), _map((1, 30))]
    view = source_view(messages, path=PATH, source=SOURCE)
    assert [(s.start, s.end) for s in view.spans] == [(1, 15), (21, 30)]
    assert not view.complete
    with pytest.raises(SourceIncomplete) as caught:
        require_complete_source(messages, path=PATH, source=SOURCE, journal=None)
    assert caught.value.missing_spans == ((16, 20),)


def test_map_without_payload_is_incomplete() -> None:
    messages = [_map((1, 10), (11, 20), (21, 30))]
    view = source_view(messages, path=PATH, source=SOURCE)
    assert not view.complete and view.spans == ()
    with pytest.raises(SourceIncomplete) as caught:
        require_complete_source(messages, path=PATH, source=SOURCE, journal=None)
    assert caught.value.missing_spans == ((1, 30),)


def test_map_at_a_stale_sha_never_completes() -> None:
    stale = "sha256:" + "0" * 64
    assert not source_view([*_three_pages(), _map((1, 30), revision=stale)],
                           path=PATH, source=SOURCE).complete


def test_map_claiming_less_than_the_file_is_incomplete() -> None:
    assert not source_view([*_three_pages(), _map((1, 20))], path=PATH, source=SOURCE).complete


@pytest.mark.parametrize("terminated", [True, False])
def test_eof_newline_is_reported_and_unterminated_last_line_verifies(terminated: bool) -> None:
    raw = SOURCE if terminated else SOURCE[:-1]
    rows = raw.decode().splitlines(keepends=True)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    pages = [_window(1, 15, source=raw), _window(16, 30, source=raw)]
    view = source_view([*pages, _map((1, 30), revision=digest)], path=PATH, source=raw)
    assert view.eof_newline is terminated
    assert view.total_lines == len(rows) == 30
    assert view.complete


def test_empty_source_view() -> None:
    view = source_view([], path=PATH, source=b"")
    assert (view.total_lines, view.eof_newline, view.spans, view.complete) == (0, False, (), False)


def test_map_round_trip_is_deterministic_and_merged() -> None:
    entries = {"b.py": ("sha256:" + "b" * 64, [(11, 20), (1, 10), (5, 12)]),
               "a.py": ("sha256:" + "a" * 64, [(3, 4)])}
    block = render_source_map(entries)
    assert block == render_source_map(dict(reversed(list(entries.items()))))
    assert block.startswith(SOURCE_MAP_HEADER + "\n")
    assert block.splitlines()[1].startswith("- a.py ") and block.endswith("spans=1-20")
    assert parse_source_map(block) == {"a.py": ("sha256:" + "a" * 64, ((3, 4),)),
                                       "b.py": ("sha256:" + "b" * 64, ((1, 20),))}
    assert render_source_map({}) is None and parse_source_map("[Host fact]\nx") is None


def test_map_with_spaces_in_path_round_trips() -> None:
    block = render_source_map({"dir with space/a b.py": (SHA, [(1, 2)])})
    assert parse_source_map(block) == {"dir with space/a b.py": (SHA, ((1, 2),))}


def _runtime(*, revision: str | None = SHA, deferred: tuple[str, ...] = (PATH,)) -> SimpleNamespace:
    plan = {"deferred_paths": list(deferred),
            "revisions": {PATH: revision} if revision is not None else {}}
    return SimpleNamespace(_request_source_closure=plan)


def _notice(path: str = PATH) -> dict[str, str]:
    return {"role": "user", "content":
            "[Deferred required source paths]\nRequired source paths are not attached; "
            f"read one only if your edit needs it:\n- {path}"}


def test_deferred_obligation_valid_stale_and_missing() -> None:
    check = lambda messages, runtime: deferred_read_obligation_valid(  # noqa: E731
        messages, path=PATH, source=SOURCE, runtime=runtime)
    assert check([_notice()], _runtime())
    assert not check([_notice()], _runtime(revision="sha256:" + "0" * 64))     # stale revision
    assert not check([_notice()], _runtime(revision=None))                     # no revision bound
    assert not check([_notice()], _runtime(deferred=()))                       # runtime forgot it
    assert not check([_notice()], SimpleNamespace())                           # no plan at all
    assert not check([], _runtime())                                           # not on the wire
    assert not check([_notice("pkg/other.py")], _runtime())                    # other path named
    assert not check([{"role": "assistant", "content": _notice()["content"]}], _runtime())


def test_file_floor_messages_price_every_page_plus_the_map_line() -> None:
    big = "".join(f"V{n} = {n}\n" for n in range(2500))
    messages = file_floor_messages("m.py", big)
    assert len(messages) == 4                       # three pages and the map line
    revision = "sha256:" + hashlib.sha256(big.encode()).hexdigest()
    parsed = parse_source_map(messages[-1]["content"])
    assert parsed == {"m.py": (revision, ((1, 2500),))}
    view = source_view(messages, path="m.py", source=big.encode())
    assert view.complete and view.total_lines == 2500


def test_deferred_obligation_reads_the_capture_frozen_plan() -> None:
    capture = SimpleNamespace(_request_source_closure_override={
        "deferred_paths": [PATH], "revisions": {PATH: SHA}})
    runtime = SimpleNamespace(_request_source_closure={"revisions": {PATH: SHA}})
    assert deferred_read_obligation_valid([_notice()], path=PATH, source=SOURCE,
                                          runtime=runtime, capture=capture)
    stale = SimpleNamespace(_request_source_closure={"revisions": {PATH: "sha256:" + "0" * 64}})
    assert not deferred_read_obligation_valid([_notice()], path=PATH, source=SOURCE,
                                              runtime=stale, capture=capture)


def test_line_grammar_matches_the_window_renderer_for_form_feed_and_nel() -> None:
    # Form feed and NEL inside a "line" are line breaks for str.splitlines and for the
    # engine's window renderer; a \n-only split would count 3 lines here, not 5.
    source = "a = 1\fb = 2\nc = 3\x85d = 4\ne = 5\n".encode()
    revision = "sha256:" + hashlib.sha256(source).hexdigest()
    rows = source_lines(source)
    assert len(rows) == 5
    assert b"".join(rows) == source
    pages = [_window(1, 2, source=source, path=PATH), _window(3, 5, source=source, path=PATH)]
    view = source_view([*pages, _map((1, 5), revision=revision)], path=PATH, source=source)
    assert view.total_lines == 5 and view.complete
    newline_only = source.split(b"\n")
    assert len([row for row in newline_only if row]) == 3  # the grammar a \n-only reader would use
