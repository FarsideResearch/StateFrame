"""B6 selection feeds B4 rendering: shapes, byte budget and frozen re-renders line up.

`relevance.select` promises a byte cost per selection (kept lines plus one marker row per
range); `render.render_view(..., frozen_selection=)` must emit exactly those bytes as the
verbatim rows of the `relevant` view, and a frozen selection must re-render byte-identically.
"""
from __future__ import annotations

from _view_ladder_fixtures import CALC_PY, artifact_for, revision_of
from kernel_view_ladder import (
    Artifact,
    Seed,
    frozen_selection,
    iter_verbatim_rows,
    omitted_ranges,
    range_marker,
    render_view,
    row_matches_slice,
    select,
    selection_bytes,
    verify_body_sha256,
)
from kernel_view_ladder import relevance as rel
from kernel_view_ladder.render import END_VIEW, format_ranges

SEEDS = (
    Seed("frame", 24, 25, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED),       # mean()
    Seed("symbol", 15, 17, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),   # Calc.add
    Seed("call", 19, 20, rel.PRIORITY_NEIGHBOUR, rel.EDGE_UNRESOLVED),    # divide, 1 gap -> merges with add
    Seed("lit", 2, 2, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),        # import json
)


def _select_for(artifact: Artifact, seeds=SEEDS, budget: int = 1 << 20):
    return select(seeds, held_text=artifact.held_text, held_start=artifact.held_start,
                  held_end=artifact.held_end, budget_bytes=budget)


def _rows_bytes(rendered_text: str) -> bytes:
    """Bytes of the verbatim rows and their markers: everything between the kept-line and [end view]."""
    body = rendered_text.split("\n", 1)[1]
    lines = body.split("\n")
    assert lines[0].startswith("kept ")
    assert lines[-1].startswith(END_VIEW)
    return ("\n".join(lines[1:-1]) + "\n").encode("utf-8", "surrogatepass")


def test_selection_shape_renders_one_marker_per_range_with_exact_omitted_record():
    artifact, _ = artifact_for("calc.py")
    selection = _select_for(artifact)
    assert selection == ((2, 2, "lit", "resolved"), (15, 20, "symbol", "resolved"), (24, 25, "frame", "observed"))
    rendered = render_view(artifact, "relevant", reason="context_limit", frozen_selection=selection)
    body = rendered.text.split("\n", 1)[1]
    markers = [line for line in body.split("\n") if line.startswith("--- lines ")]
    assert markers == [range_marker(a, b, f"{kind}, {edge}") for a, b, kind, edge in selection]
    omitted = omitted_ranges(selection, artifact.held_start, artifact.held_end)
    assert body.split("\n")[0] == (f"kept 9 of 29 lines in 3 ranges; omitted: {format_ranges(omitted)}")
    rows = list(iter_verbatim_rows(rendered.text))
    assert [(a, b) for a, b, _ in rows] == [(a, b) for a, b, _, _ in selection]
    for a, b, text in rows:
        assert row_matches_slice(text, artifact.slice(a, b))
    assert verify_body_sha256(rendered.text)


def test_selection_bytes_equal_rendered_row_bytes_and_budget_is_tight():
    artifact, _ = artifact_for("calc.py")
    selection = _select_for(artifact)
    promised = selection_bytes(selection, artifact.held_text, artifact.held_start)
    rendered = render_view(artifact, "relevant", reason="context_limit", frozen_selection=selection)
    assert len(_rows_bytes(rendered.text)) == promised
    # The promise is tight: exactly that budget admits the selection, one byte less does not.
    assert _select_for(artifact, budget=promised) == selection
    smaller = _select_for(artifact, budget=promised - 1)
    assert smaller != selection
    if smaller is not None:
        again = render_view(artifact, "relevant", reason="context_limit", frozen_selection=smaller)
        assert len(_rows_bytes(again.text)) == selection_bytes(smaller, artifact.held_text, artifact.held_start)


def test_unterminated_last_line_and_crlf_are_budgeted_as_rendered():
    text = "a = 'é'\r\nb = 2\r\nc = 3\r\nd = 4\r\ne = 5\r\nf = 6"  # file ends without a newline
    artifact = Artifact(display="x.py", revision=revision_of(text.encode()), held_start=1, held_end=6,
                        total_lines=6, held_text=text)
    # Realistic selections only: `select` already merged anything within MERGE_GAP, so the ranges
    # here are at least three omitted lines apart (the renderer would otherwise merge them).
    for selection in (((1, 1, "lit", "resolved"), (6, 6, "frame", "observed")),
                      ((5, 6, "symbol", "resolved"),), ((1, 2, "lit", "resolved"),)):
        rendered = render_view(artifact, "relevant", reason="model_keep", frozen_selection=selection)
        assert len(_rows_bytes(rendered.text)) == selection_bytes(selection, text, 1)


def test_frozen_selection_re_renders_byte_identically_across_requests():
    artifact, _ = artifact_for("calc.py")
    state: dict[object, dict] = {}
    key = ("calc.py", artifact.revision, artifact.held_start, artifact.held_end)
    calls = []

    def compute():
        calls.append(1)
        return _select_for(artifact)

    first = frozen_selection(state, key, compute)
    second = frozen_selection(state, key, lambda: (("never", 0, 0, "x"),))  # must not be consulted
    assert first == second and calls == [1]
    render_a = render_view(artifact, "relevant", reason="context_limit", frozen_selection=first)
    render_b = render_view(artifact, "relevant", reason="context_limit", frozen_selection=tuple(reversed(second)))
    assert render_a.text.encode() == render_b.text.encode()
    assert render_a.vid == render_b.vid
    # A different selection or reason is a different view id, so the stability is not vacuous.
    assert render_view(artifact, "relevant", reason="model_keep", frozen_selection=first).vid != render_a.vid
    assert render_view(artifact, "relevant", reason="context_limit", frozen_selection=first[:1]).vid != render_a.vid


def test_relevance_merge_gap_matches_renderer_merge_gap():
    # Both sides merge ranges separated by at most two omitted lines; a third omitted line keeps them apart.
    assert rel.MERGE_GAP == 2
    from kernel_view_ladder.render import MERGE_GAP_LINES, normalize_ranges
    assert MERGE_GAP_LINES == rel.MERGE_GAP
    artifact, _ = artifact_for("calc.py")
    seeds = (Seed("lit", 2, 2, 4, rel.EDGE_RESOLVED), Seed("lit", 5, 5, 4, rel.EDGE_RESOLVED),
             Seed("lit", 9, 9, 4, rel.EDGE_RESOLVED))
    selection = _select_for(artifact, seeds)
    assert selection == ((2, 5, "lit", "resolved"), (9, 9, "lit", "resolved"))
    # The renderer leaves an already-merged selection alone: same ranges, same labels.
    assert normalize_ranges(selection, 1, 29) == ((2, 5, "lit, resolved"), (9, 9, "lit, resolved"))
    assert CALC_PY.splitlines()[1].startswith("import json")
