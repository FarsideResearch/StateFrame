"""kernel_edit_handles: deterministic request-scoped handles, a positive [CURRENT SOURCE]
map with explicit gaps, final-user-message placement, and no footprint anywhere else."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from kernel_edit_handles import (  # noqa: E402
    CURRENT_SOURCE_PREFIX, HANDLE_PATTERN, DeliveredSpan, HandleMap, SpanKind, append_map_to_final_user_message,
    handle_name, line_byte_range, line_gaps, mint_handles, parse_handle, parse_map,
    spans_from_open_sites,
)
from kernel_edit_refusal_codes import HANDLE_PATTERN as CATALOG_HANDLE_PATTERN  # noqa: E402
from kernel_frame_optional_allocation import site_identity  # noqa: E402
from kernel_paged_source import source_revision  # noqa: E402

SOURCE = "".join(f"line {number}\n" for number in range(1, 41))
OTHER = "alpha\r\nbeta\r\ngamma\r\n"
PATH = "src/pkg/module.py"
OTHER_PATH = "src/pkg/other.py"


def _spans() -> list[DeliveredSpan]:
    return [
        DeliveredSpan.from_source(PATH, SOURCE, start_line=1, end_line=10),
        DeliveredSpan.from_source(PATH, SOURCE, start_line=20, end_line=30),
        DeliveredSpan.from_source(OTHER_PATH, OTHER, kind=SpanKind.REWRITE),
    ]


# --- minting -----------------------------------------------------------------------

def test_handle_names_follow_the_catalog_pattern_and_round_trip():
    assert HANDLE_PATTERN.pattern == CATALOG_HANDLE_PATTERN.pattern
    assert handle_name(42, 1) == "q42h1"
    assert HANDLE_PATTERN.fullmatch(handle_name(0, 7))
    assert parse_handle("q42h3") == (42, 3)
    assert parse_handle("source-range:a:b") is None and parse_handle(7) is None
    with pytest.raises(ValueError):
        handle_name(-1, 1)
    with pytest.raises(ValueError):
        handle_name(1, 0)
    with pytest.raises(ValueError):
        handle_name(True, 1)  # bools are not ordinals


def test_minting_is_deterministic_from_request_ordinal_and_span_order():
    first = mint_handles(7, _spans())
    again = mint_handles(7, _spans())
    assert first == again
    assert [record.handle for record in first.records] == ["q7h1", "q7h2", "q7h3"]
    assert first.render() == again.render()
    assert first.as_dict() == again.as_dict()
    # Replay re-mints identically from (ordinal, spans); the next request renames only
    # the request prefix, never the per-span ordinal.
    later = mint_handles(8, _spans())
    assert [record.handle for record in later.records] == ["q8h1", "q8h2", "q8h3"]
    assert [record.span for record in later.records] == [record.span for record in first.records]
    reordered = mint_handles(7, list(reversed(_spans())))
    assert reordered.lookup("q7h1").path == OTHER_PATH


def test_records_carry_path_revision_bytes_and_kind():
    handle_map = mint_handles(3, _spans())
    first = handle_map.lookup("q3h1")
    assert first.path == PATH and first.revision == source_revision(SOURCE)
    assert (first.span.byte_start, first.span.byte_end) == (0, len("".join(f"line {n}\n" for n in range(1, 11))))
    assert first.kind is SpanKind.SPAN
    assert first.text(SOURCE) == "".join(f"line {n}\n" for n in range(1, 11))
    assert first.covers_bytes(0, 5) and first.covers_bytes(first.span.byte_end - 1, first.span.byte_end)
    assert not first.covers_bytes(first.span.byte_end - 1, first.span.byte_end + 1)
    rewrite = handle_map.lookup("q3h3")
    assert rewrite.kind is SpanKind.REWRITE
    assert rewrite.span.byte_end == len(OTHER.encode("utf-8"))
    assert rewrite.text(OTHER) == OTHER  # CRLF preserved byte for byte
    with pytest.raises(ValueError):
        first.text(SOURCE + "tail\n")
    assert handle_map.lookup("q3h9") is None
    row = first.as_dict()
    assert row["handle"] == "q3h1" and row["kind"] == "SPAN" and row["total_lines"] == 40


def test_byte_offsets_count_utf8_bytes_not_characters():
    source = "é\n" + "x\n" * 3
    byte_start, byte_end, total = line_byte_range(source, 2, 2)
    assert (byte_start, byte_end, total) == (3, 5, 4)
    with pytest.raises(ValueError):
        line_byte_range(source, 0, 1)
    with pytest.raises(ValueError):
        line_byte_range(source, 2, 1)
    with pytest.raises(ValueError):
        line_byte_range(source, 1, 5)


def test_minting_rejects_duplicates_mixed_revisions_and_partial_rewrites():
    with pytest.raises(ValueError, match="duplicate"):
        mint_handles(1, [_spans()[0], _spans()[0]])
    changed = DeliveredSpan.from_source(PATH, SOURCE + "more\n", start_line=1, end_line=5)
    with pytest.raises(ValueError, match="revision"):
        mint_handles(1, [_spans()[0], changed])
    with pytest.raises(ValueError, match="whole file"):
        DeliveredSpan.from_source(PATH, SOURCE, start_line=1, end_line=5, kind=SpanKind.REWRITE)
    with pytest.raises(ValueError):
        DeliveredSpan.from_source(PATH, "", kind=SpanKind.REWRITE)
    with pytest.raises(ValueError):
        mint_handles(-1, [])
    with pytest.raises(ValueError):
        mint_handles(1, [{"path": PATH}])  # type: ignore[list-item]


# --- the map -----------------------------------------------------------------------

def test_map_lists_every_delivered_span_and_states_gaps():
    handle_map = mint_handles(7, _spans())
    text = handle_map.render()
    assert text.startswith(CURRENT_SOURCE_PREFIX + "\n")
    assert text.count(CURRENT_SOURCE_PREFIX) == 1
    for record in handle_map.records:
        assert f"{record.handle} lines {record.span.start_line}-{record.span.end_line}" in text
    assert f"file {PATH} revision {source_revision(SOURCE)} (40 lines)" in text
    assert "  not shown: lines 11-19, 31-40" in text
    assert f"file {OTHER_PATH} revision {source_revision(OTHER)} (3 lines)" in text
    assert "q7h3 lines 1-3 (rewrite, whole file)" in text
    assert "  not shown: none; the whole file is delivered" in text
    # Positive statement for the model: handles are the only edit surface; no bodies.
    assert "cannot be edited through these handles" in text
    assert "line 1\n" not in text and "alpha" not in text
    parsed = parse_map("preamble\n\n" + text)
    assert parsed["request_ordinal"] == 7
    assert [row["path"] for row in parsed["files"]] == [PATH, OTHER_PATH]
    assert parsed["files"][0]["not_shown"] == [(11, 19), (31, 40)]
    assert parsed["files"][0]["handles"][1] == {
        "handle": "q7h2", "start_line": 20, "end_line": 30, "kind": "SPAN"}
    assert parsed["files"][1]["handles"][0]["kind"] == "REWRITE"
    assert parsed["files"][1]["not_shown"] == []
    assert parse_map("no map here") is None and parse_map(None) is None


def test_gap_arithmetic_merges_adjacent_and_overlapping_spans():
    spans = [
        DeliveredSpan.from_source(PATH, SOURCE, start_line=5, end_line=10),
        DeliveredSpan.from_source(PATH, SOURCE, start_line=11, end_line=12),
        DeliveredSpan.from_source(PATH, SOURCE, start_line=8, end_line=15),
        DeliveredSpan.from_source(PATH, SOURCE, start_line=39, end_line=39),
    ]
    assert line_gaps(spans) == [(1, 4), (16, 38), (40, 40)]
    assert mint_handles(1, spans).render().count("not shown: lines 1-4, 16-38, 40") == 1
    assert line_gaps([]) == []
    whole = [DeliveredSpan.from_source(PATH, SOURCE)]
    assert line_gaps(whole) == []
    with pytest.raises(ValueError):
        line_gaps([spans[0], DeliveredSpan.from_source(PATH, SOURCE + "x\n", start_line=1, end_line=2)])


def test_empty_map_says_nothing_was_delivered():
    empty = mint_handles(4, [])
    assert empty.records == () and empty.paths() == ()
    text = empty.render()
    assert text.startswith(CURRENT_SOURCE_PREFIX)
    assert "0 handles over 0 files" in text and "read before editing" in text
    assert parse_map(text) == {"request_ordinal": 4, "files": []}


# --- placement: final user message only, prefix untouched ------------------------------

def _messages(turn: int) -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": f"thinking {turn}"},
        {"role": "user", "content": f"observation {turn}\n"},
    ]


def test_map_is_appended_to_the_final_user_message_only():
    handle_map = mint_handles(7, _spans())
    messages = _messages(7)
    before = copy.deepcopy(messages)
    placed = append_map_to_final_user_message(messages, handle_map)
    assert messages == before  # input untouched
    assert placed[:-1] == messages[:-1]
    assert all(a is b for a, b in zip(placed[:-1], messages[:-1]))  # same objects: prefix cache-safe
    assert placed[-1]["role"] == "user"
    assert placed[-1]["content"] == "observation 7\n\n" + handle_map.render()
    assert sum(CURRENT_SOURCE_PREFIX in str(m["content"]) for m in placed) == 1
    joined = json.dumps(placed[:-1])
    assert HANDLE_PATTERN.search(joined) is None


def test_consecutive_requests_share_the_retained_prefix_and_differ_only_in_the_map():
    first = append_map_to_final_user_message(_messages(7), mint_handles(7, _spans()))
    second_history = [*first, {"role": "assistant", "content": "edit"},
                      {"role": "user", "content": "observation 8"}]
    second = append_map_to_final_user_message(second_history, mint_handles(8, _spans()))
    # Request N's map was in N's final user message; in N+1 that message is retained as
    # it was sent, so the shared prefix is byte-identical and N+1 adds exactly one map.
    assert second[: len(first)] == first
    assert second[-1]["content"].endswith(mint_handles(8, _spans()).render())
    assert "q7h1" in json.dumps(first[-1]) and "q8h1" in json.dumps(second[-1])
    assert "q8h1" not in json.dumps(second[:-1])


def test_placement_refuses_wrong_tail_and_double_maps():
    handle_map = mint_handles(1, _spans())
    with pytest.raises(ValueError):
        append_map_to_final_user_message([], handle_map)
    with pytest.raises(ValueError):
        append_map_to_final_user_message([{"role": "assistant", "content": "x"}], handle_map)
    with pytest.raises(ValueError):
        append_map_to_final_user_message([{"role": "user", "content": [{"type": "text"}]}], handle_map)
    once = append_map_to_final_user_message([{"role": "user", "content": "x"}], handle_map)
    with pytest.raises(ValueError, match="already"):
        append_map_to_final_user_message(once, handle_map)
    bare = append_map_to_final_user_message([{"role": "user", "content": ""}], handle_map)
    assert bare[-1]["content"] == handle_map.render()


# --- open sites: read-only derivation, inventory identity unchanged ------------------------

def _site(path: str, source: str, start: int, end: int, **overrides: object) -> dict[str, object]:
    lines = source.splitlines(keepends=True)
    excerpt = "".join(lines[start - 1:end])
    site: dict[str, object] = {
        "schema": "stateframe-framefit4-open-edit-site/1",
        "source": "last_read_or_edit_region",
        "literal": "",
        "path": path,
        "line": start,
        "lines": {"start": start, "end": end, "total": len(lines)},
        "revision": source_revision(source),
        "content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "current_bytes": excerpt,
        "current_line": lines[start - 1].rstrip("\n"),
        "read_this_request": True,
    }
    site.update(overrides)
    return site


def test_handles_mode_does_not_change_site_inventory_identity_across_requests():
    sites = [_site(PATH, SOURCE, 1, 10), _site(PATH, SOURCE, 20, 30),
             _site(OTHER_PATH, OTHER, 1, 3)]
    sources = {PATH: SOURCE, OTHER_PATH: OTHER}
    frozen = copy.deepcopy(sites)
    identities = [site_identity(site) for site in sites]

    spans = spans_from_open_sites(sites, sources)
    request_n = mint_handles(11, spans)
    request_n1 = mint_handles(12, spans_from_open_sites(sites, sources))
    append_map_to_final_user_message(_messages(11), request_n)
    append_map_to_final_user_message(_messages(12), request_n1)

    assert sites == frozen  # rows never mutated
    assert [site_identity(site) for site in sites] == identities
    assert [r.handle for r in request_n.records] == ["q11h1", "q11h2", "q11h3"]
    assert [r.handle for r in request_n1.records] == ["q12h1", "q12h2", "q12h3"]
    assert [r.span for r in request_n.records] == [r.span for r in request_n1.records]
    # The map never becomes a site field: nothing in any row spells a handle or the block.
    serialized = json.dumps(sites, sort_keys=True)
    assert HANDLE_PATTERN.search(serialized) is None
    assert CURRENT_SOURCE_PREFIX not in serialized
    # Per-request delivery annotations (the carrier ed2f512e excludes) do not alter the
    # spans that back the handles either: a site shown as not-read is simply skipped.
    hidden = copy.deepcopy(sites)
    hidden[1]["read_this_request"] = False
    hidden[1]["current_bytes"] = None
    assert [site_identity(site) for site in hidden] == identities
    assert [s.start_line for s in spans_from_open_sites(hidden, sources)] == [1, 1]


def test_spans_from_open_sites_refuses_stale_or_malformed_rows():
    sources = {PATH: SOURCE}
    with pytest.raises(ValueError, match="revision"):
        spans_from_open_sites([_site(PATH, SOURCE + "x\n", 1, 2)], sources)
    with pytest.raises(ValueError, match="no current source"):
        spans_from_open_sites([_site(OTHER_PATH, OTHER, 1, 1)], sources)
    with pytest.raises(ValueError, match="lines"):
        spans_from_open_sites([{"path": PATH}], sources)
    with pytest.raises(ValueError, match="integer line span"):
        spans_from_open_sites([_site(PATH, SOURCE, 1, 2, lines={"start": "1", "end": 2})], sources)
    assert spans_from_open_sites([], sources) == ()


# --- footprint: schemas carry no handles, nothing imports the module, no env reads -----------

def test_tool_schema_json_never_spells_a_handle():
    from probes import react_repair

    menus = [react_repair.TOOLS]
    for gate in (False, True):
        for probe in (False, True):
            menus.append(react_repair._tools_for_probe_modes(
                probe_gate_on=gate, observation_probe_on=probe))
    for menu in menus:
        assert menu, "tool menu is empty"
        serialized = json.dumps(menu, sort_keys=True)
        assert HANDLE_PATTERN.search(serialized) is None
        assert CURRENT_SOURCE_PREFIX not in serialized
        assert "edit_source" not in serialized and "rewrite_source" not in serialized


def test_only_the_protocol_executor_imports_the_module():
    needle = re.compile(r"^\s*(from|import)\s+kernel_edit_handles\b", re.MULTILINE)
    importers = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "src").rglob("*.py")
        if needle.search(path.read_text(encoding="utf-8"))
    )
    # Edit protocol modules only; legacy paths reach it solely through the
    # handles-gated send-time placement (kernel_edit_request_map).
    assert importers == ["src/stateframe/engine/kernel_edit_repair_hook.py",
                         "src/stateframe/engine/kernel_edit_request_map.py",
                         "src/stateframe/engine/kernel_edit_source_tool.py"]


def test_module_is_pure_no_environment_or_io():
    text = (ENGINE / "kernel_edit_handles.py").read_text(encoding="utf-8")
    assert "os.environ" not in text and "import os" not in text
    assert "open(" not in text and "subprocess" not in text and "time." not in text
    assert "STATEFRAME_EDIT_PROTOCOL" not in text  # the switch is wired in a later unit
    assert isinstance(mint_handles(0, []), HandleMap)
