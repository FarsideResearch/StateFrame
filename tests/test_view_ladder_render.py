"""View ladder renderer: wire grammar and the spec's worked examples (compressed_views_spec.md 1.2)."""
from __future__ import annotations

import pytest

from _view_ladder_fixtures import CALC_PY, FIXTURES, artifact_for, revision_of
from kernel_view_ladder import (
    READ_VIEW_HEADER,
    RUNGS,
    Artifact,
    applicable_rungs,
    change_notice,
    enabled,
    is_degraded_view,
    iter_verbatim_rows,
    parse_view_header,
    render_stub_line,
    render_view,
    row_matches_slice,
    verify_body_sha256,
)
from kernel_view_ladder.render import VIEW_LADDER_ENV, normalize_ranges, complement

CALC_REV = revision_of(CALC_PY.encode())


def _body(rendered) -> str:
    return rendered.text.split("\n", 1)[1]


def test_switch_default_off_and_lenient():
    assert enabled({}) is False
    assert enabled({VIEW_LADDER_ENV: " On "}) is True
    assert enabled({VIEW_LADDER_ENV: "1"}) is False


def test_is_degraded_view_only_for_read_view_prefix():
    assert is_degraded_view("[read view] view=stub path=x")
    assert not is_degraded_view("[read page] path=x revision=sha256:aa lines=1-2/2")
    assert not is_degraded_view(None)
    assert not is_degraded_view("[read view]")  # the prefix includes the space


@pytest.mark.parametrize("rung", ("range", "relevant", "symbols", "outline", "stub"))
def test_header_grammar_fixed_key_order(rung):
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, rung, reason="context_limit", ranges=[(15, 20), (24, 25)],
                           frozen_selection=[(15, 20, "frame", "observed")])
    first = rendered.text.split("\n", 1)[0]
    match = READ_VIEW_HEADER.match(first)
    assert match, first
    assert match["view"] == rung and match["path"] == "pkg/calc.py" and match["revision"] == CALC_REV
    assert (match["start"], match["end"], match["total"]) == ("1", "29", "29")
    assert match["reason"] == "context_limit"
    assert match["vid"] == rendered.vid and match["body_sha256"] == rendered.body_sha256
    header = parse_view_header(rendered.text)
    assert header and header["start"] == 1 and header["end"] == 29 and header["total"] == 29
    assert verify_body_sha256(rendered.text)
    assert "\r" not in first
    # No volatile facts anywhere in the body.
    for token in ("turn=", "@t", "request_id", "/home/", "age="):
        assert token not in _body(rendered)


def test_full_is_not_rendered_here_and_unknown_reason_rejected():
    artifact, _ = artifact_for("calc.py")
    with pytest.raises(ValueError):
        render_view(artifact, "full", reason="context_limit")
    with pytest.raises(ValueError):
        render_view(artifact, "stub", reason="bored")


def test_range_view_matches_spec_example():
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, "relevant", reason="context_limit", frozen_selection=[(24, 25), (15, 20)])
    expected_body = (
        "kept 8 of 29 lines in 2 ranges; omitted: 1-14, 21-23, 26-29\n"
        "--- lines 15-20 ---\n"
        "    def add(self, value: int) -> int:\n"
        "        self.total += value * self.scale\n"
        "        return self.total\n"
        "\n"
        "    def divide(self, value: int) -> float:\n"
        "        return self.total / value\n"
        "--- lines 24-25 ---\n"
        "    return sum(values) / len(values)\n"
        "\n"
        '[end view] re-read for full: read(path="pkg/calc.py", start_line=1, end_line=29)'
    )
    assert _body(rendered) == expected_body
    rows = list(iter_verbatim_rows(rendered.text))
    assert [(s, e) for s, e, _ in rows] == [(15, 20), (24, 25)]
    for start, end, text in rows:
        assert row_matches_slice(text, artifact.slice(start, end))


def test_range_normalisation_sorts_clamps_merges_and_labels():
    merged = normalize_ranges([(30, 24, "lit"), (12, 10), (3, 5, "frame", "observed"), (100, 200)], 1, 29)
    assert merged == ((3, 5, "frame, observed"), (10, 12, ""), (24, 29, "lit"))
    assert normalize_ranges([(1, 4), (7, 9)], 1, 20) == ((1, 9, ""),)  # gap of two lines merges
    assert normalize_ranges([(1, 4), (8, 9)], 1, 20) == ((1, 4, ""), (8, 9, ""))
    assert complement(((3, 5, ""), (10, 12, "")), 1, 12) == [(1, 2), (6, 9)]
    assert complement(((1, 12, ""),), 1, 12) == []


def test_relevant_edge_label_in_marker():
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, "relevant", reason="context_limit", frozen_selection=[(24, 25, "frame", "observed")])
    assert "--- lines 24-25 (frame, observed) ---" in rendered.text


def test_range_view_needs_a_subrange():
    artifact, _ = artifact_for("calc.py")
    with pytest.raises(ValueError):
        render_view(artifact, "range", reason="model_keep", ranges=[(40, 50)])


def test_symbols_view_matches_spec_example():
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, "symbols", reason="context_limit")
    expected_body = (
        "parser=ast\n"
        "L2      import json\n"
        "L3      from typing import Sequence\n"
        "L5      DEFAULT_SCALE = 2\n"
        "L8-20   class Calc:\n"
        "L11-13    def __init__(self, scale: int = DEFAULT_SCALE) -> None:\n"
        "L15-17    def add(self, value: int) -> int:\n"
        "L19-20    def divide(self, value: int) -> float:\n"
        "L23-24  def mean(values: Sequence[int]) -> float:\n"
        "L27-28  def dump(calc: Calc) -> str:\n"
        "syntactic: call=sum at=24; call=len at=24; call=json.dumps at=28\n"
        "omitted: bodies of 6 definitions (lines 12-13, 16-17, 20, 24, 28), docstrings at 1, 9\n"
        '[end view] bodies omitted; re-read a symbol with read(path="pkg/calc.py", symbol="Calc.add") '
        "or a range with start_line/end_line"
    )
    assert _body(rendered) == expected_body


def test_outline_view_is_definition_rows_only():
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, "outline", reason="model_keep")
    assert _body(rendered) == (
        "parser=ast\n"
        "L8-20   class Calc:\n"
        "L11-13    def __init__(self, scale: int = DEFAULT_SCALE) -> None:\n"
        "L15-17    def add(self, value: int) -> int:\n"
        "L19-20    def divide(self, value: int) -> float:\n"
        "L23-24  def mean(values: Sequence[int]) -> float:\n"
        "L27-28  def dump(calc: Calc) -> str:\n"
        "omitted: bodies\n"
        '[end view] bodies omitted; re-read a symbol with read(path="pkg/calc.py", symbol="Calc.add") '
        "or a range with start_line/end_line"
    )


def test_stub_matches_spec_shape_and_shares_retention_fields():
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, "stub", reason="context_limit")
    line = render_stub_line(display="pkg/calc.py", revision=CALC_REV, held_start=1, held_end=29, total_lines=29,
                            reason="context_limit")
    assert _body(rendered) == line
    assert line == (
        f"READ_RESULT_DEGRADED path=pkg/calc.py sha=sha256:{CALC_REV[7:19]} lines=1-29/29 "
        'full content omitted (context limit); re-read for full: read(path="pkg/calc.py", start_line=1, end_line=29)'
    )
    assert "READ_RESULT_DEGRADED" in line and " sha=sha256:" in line and " lines=1-29/29" in line


def test_reason_is_part_of_bytes_and_vid():
    artifact, _ = artifact_for("calc.py")
    a = render_view(artifact, "stub", reason="context_limit")
    b = render_view(artifact, "stub", reason="model_keep")
    assert a.vid != b.vid and a.text != b.text
    assert "(model keep)" in b.text


def test_diet_identity_grammar_still_parses_the_view_header():
    """`_READ_PATH_PATTERN` / `_SOURCE_RANGE_PATTERNS` from kernel_request_diet classify the group."""
    from kernel_request_diet import _READ_PATH_PATTERN, _SOURCE_RANGE_PATTERNS, _REVISION_PATTERN

    artifact, _ = artifact_for("calc.py")
    text = render_view(artifact, "symbols", reason="context_limit").text
    first = text.split("\n", 1)[0]
    assert _READ_PATH_PATTERN.search(first).group(1) == "pkg/calc.py"
    assert _REVISION_PATTERN.search(first).group(1) == CALC_REV
    hits = [p.search(first) for p in _SOURCE_RANGE_PATTERNS]
    assert any(h and (h.group(1), h.group(2)) == ("1", "29") for h in hits)
    from kernel_request_source_window import _READ_PAGE_HEADER
    assert _READ_PAGE_HEADER.match(first) is None


def test_change_notice_shape_and_absence():
    artifact, text = artifact_for("calc.py")
    assert change_notice(artifact, disk_revision=CALC_REV, disk_text=text) is None  # P10: never empty
    new_text = text.replace("        return self.total\n", "        self.calls += 1\n        return self.total\n")
    new_rev = revision_of(new_text.encode())
    notice = change_notice(artifact, disk_revision=new_rev, disk_text=new_text)
    assert notice == (f"changed: file is now @{new_rev[7:15]}; held 1-29 maps to 1-30 (+1 -0 lines inside range); "
                      "re-read before relying on this view")
    rendered = render_view(artifact, "symbols", reason="context_limit", disk_revision=new_rev, disk_text=new_text)
    body_lines = _body(rendered).split("\n")
    assert body_lines[-2] == notice and body_lines[-1].startswith("[end view]")
    assert "+" not in body_lines[-2].replace("(+1 -0", "") and "self.calls" not in rendered.text  # no hunk bodies


def test_change_notice_held_subrange_ignores_content_outside_it():
    artifact, text = artifact_for("calc.py", 15, 20)
    new_text = "# new header\n" + text + "\n# trailer\n"
    notice = change_notice(artifact, disk_revision=revision_of(new_text.encode()), disk_text=new_text)
    assert "held 15-20 maps to 16-21 (+0 -0 lines inside range)" in notice


def test_change_notice_when_range_deleted():
    artifact, text = artifact_for("calc.py", 15, 20)
    new_text = "x = 1\n"
    notice = change_notice(artifact, disk_revision=revision_of(new_text.encode()), disk_text=new_text)
    assert "maps to none (+0 -6 lines inside range)" in notice


def test_applicable_rungs_follow_content_rule():
    calc, _ = artifact_for("calc.py")
    assert applicable_rungs(calc) == ("range", "relevant", "symbols", "outline", "stub")
    empty, _ = artifact_for("empty.py")
    assert applicable_rungs(empty) == ("stub",)
    binary, _ = artifact_for("binary.bin")
    assert applicable_rungs(binary) == ("range", "relevant", "stub")
    one, _ = artifact_for("one_line_no_newline.txt")
    assert "symbols" not in applicable_rungs(one)


def test_artifact_rejects_line_count_mismatch():
    with pytest.raises(ValueError):
        Artifact(display="x.py", revision=CALC_REV, held_start=1, held_end=3, total_lines=3, held_text="a\nb\n")


def test_rungs_constant_order():
    assert RUNGS == ("full", "range", "relevant", "symbols", "outline", "stub")
    assert set(FIXTURES) >= {"calc.py", "worker.log", "binary.bin"}
