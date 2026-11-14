"""View ladder pitfalls P1, P2, P4, P5, P10, P11 (compressed_views_spec.md 1.9) plus structure maps."""
from __future__ import annotations

import random

import pytest

from _view_ladder_fixtures import FIXTURES, artifact_for, revision_of
from kernel_view_ladder import (
    applicable_rungs,
    candidate_rungs,
    change_notice,
    iter_verbatim_rows,
    pick_rung,
    render_view,
    row_matches_slice,
)
from kernel_view_ladder import structure
from kernel_view_ladder.render import Artifact


def _renders(artifact, full_text: str, **kwargs) -> dict[str, str]:
    out = {"full": full_text}
    for rung in applicable_rungs(artifact):
        extra = {}
        if rung == "range":
            extra["ranges"] = kwargs.get("ranges", [(artifact.held_start, artifact.held_start)])
        if rung == "relevant":
            extra["frozen_selection"] = kwargs.get("frozen_selection", [(artifact.held_start, artifact.held_start, "frame", "observed")])
        out[rung] = render_view(artifact, rung, reason="context_limit", **extra).text
    return out


def _page_text(artifact) -> str:
    """A stand-in for today's `[read page]` carrier: header line plus the held text."""
    return (f"[read page] path={artifact.display} revision={artifact.revision} "
            f"lines={artifact.held_start}-{artifact.held_end}/{artifact.total_lines} requested=x effective=lines:"
            f"{artifact.held_start}-{artifact.held_end}@0:0 clip=none chars=0 bytes=0 truncated=false "
            f"decoded_with_replacement=false\n" + artifact.held_text)


# --- P1: scoped to the held range ---------------------------------------------------------------

def test_structure_view_scoped_to_held_range():
    artifact, text = artifact_for("nested.py", 15, 22)  # inside Outer.Inner.run
    outline = structure.outline_for(artifact, file_text=text)
    assert outline is not None
    enclosing = [row for row in outline.rows if row.enclosing]
    inside = [row for row in outline.rows if not row.enclosing]
    assert enclosing and len(enclosing) <= 3  # <= nesting depth (Outer > Inner > run)
    assert all(artifact.held_start <= row.start <= artifact.held_end for row in inside)
    rendered = render_view(artifact, "symbols", reason="context_limit", file_text=text).text
    lines = rendered.split("\n")
    assert lines[2].startswith("L11") and "(enclosing, starts L11)" in lines[2]  # class Outer:
    assert "(enclosing, starts L14)" in lines[3]  # class Inner:
    assert lines[4].startswith("L15-22      def run( self, count: int, ) -> int:")  # starts inside: a normal row
    assert "import=sys at=19" in rendered and "raise=ValueError at=21" in rendered and "call=os.getpid at=22" in rendered
    assert "L33" not in rendered  # nothing from outside the held range except enclosing headers


def test_structure_view_without_file_text_uses_dedent_retry():
    artifact, _ = artifact_for("indented_slice.py")
    outline = structure.outline_for(artifact)
    assert outline.parser == "ast"
    assert [row.text for row in outline.rows] == ["def inner(self):", "def other(self):"]
    assert [row.start for row in outline.rows] == [1, 4]


def test_structure_syntax_error_falls_back_to_scan():
    artifact, _ = artifact_for("syntax_error.py")
    outline = structure.outline_for(artifact)
    assert outline.parser == "scan"
    assert any(row.text.startswith("class Ok") for row in outline.rows)


def test_decorators_and_nested_definitions_render_with_anchors():
    artifact, text = artifact_for("nested.py")
    rendered = render_view(artifact, "symbols", reason="context_limit").text
    assert "L10     @dataclass" in rendered
    assert "L11-28  class Outer:" in rendered
    assert "L14-22    class Inner:" in rendered
    assert "L15-22      def run( self, count: int, ) -> int:" in rendered
    assert "L24       @staticmethod" in rendered
    assert "L26-27      def helper():" in rendered
    assert "L31-32  async def main() -> None:" in rendered
    assert "L2-5    from dataclasses import ( dataclass, field, )" in rendered
    assert "L7      LIMIT = 3" in rendered
    assert 'symbol="Outer.Inner.run"' in rendered


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_structure_rows_are_slices_of_the_held_lines(name):
    artifact, _ = artifact_for(name)
    outline = structure.outline_for(artifact)
    if outline is None:
        return
    for row in outline.rows:
        if row.enclosing or row.kind in ("key", "note", "block"):
            continue
        end = min(row.header_end if row.header_end is not None else (row.end or row.start), artifact.held_end)
        joined = " ".join(l.strip() for l in artifact.slice(row.start, end).splitlines())
        first = artifact.line(row.start).strip()
        if row.kind == "log":
            assert row.text.startswith(f"L{row.start} ")
            assert first.startswith(row.text.split(" ", 1)[1].split(" (x")[0])
        elif row.kind == "constant":
            assert row.text == first
        elif outline.parser == "scan":
            assert first.startswith(row.text) or joined.startswith(row.text)
        else:
            assert row.text == joined, (name, row)


def test_non_python_maps_have_line_anchors_and_kinds():
    js, _ = artifact_for("app.js")
    text = render_view(js, "symbols", reason="context_limit").text
    assert "parser=scan" in text and "L4-6    export function load(name) {" in text
    assert "L8-10   const handler = async (req) => {" in text and "L12-16  export default class Server {" in text
    assert "L13-15    start(port) {" in text
    go, _ = artifact_for("main.go")
    text = render_view(go, "outline", reason="context_limit").text
    assert "L5-7    type Server struct {" in text and "L9-12   func (s *Server) Start() error {" in text
    rust, _ = artifact_for("lib.rs")
    text = render_view(rust, "symbols", reason="context_limit").text
    assert "L7-11   impl fmt::Display for Point {" in text and "L8-10     fn fmt(" in text
    assert "L13-15  pub(crate) fn origin() -> Point {" in text
    assert 'symbol="Point"' in text


def test_data_file_maps():
    pkg, _ = artifact_for("package.json")
    text = render_view(pkg, "symbols", reason="context_limit").text
    assert "parser=json" in text
    assert "long: str(40)" in text and "nested: map[3]" in text and "  deep: map[1]" in text
    assert "tags: list[2] of str" in text and "version: int" in text
    toml, _ = artifact_for("pyproject.toml")
    text = render_view(toml, "symbols", reason="context_limit").text
    assert "parser=tomllib" in text and "project: map[2]" in text and "  pytest: map[1]" in text
    yaml, _ = artifact_for("compose.yaml")
    text = render_view(yaml, "symbols", reason="context_limit").text
    assert "L1      name: scalar" in text and "L2      services: map" in text and "L3        web: map" in text
    assert "L7      ports: list[2]" in text and "L11     empty: map" in text
    md, _ = artifact_for("README.md")
    text = render_view(md, "outline", reason="context_limit").text
    assert "L1      # Title" in text and "L5        ## Section A" in text and "L9          ### Sub" in text
    assert "L13-14  Setext" in text


def test_log_map_head_tail_errors_collapsed():
    log, _ = artifact_for("worker.log")
    text = render_view(log, "symbols", reason="context_limit").text
    lines = text.split("\n")
    assert "parser=log" in text
    assert "L1-10   head" in text and "L2981-3000 tail" in text
    assert "L1 2026-01-01 00:00:01 INFO tick 1" in text
    errors_index = next(i for i, line in enumerate(lines) if line.endswith(" errors"))
    error_rows = [line for line in lines[errors_index + 1:] if " ERROR " in line or "Traceback" in line]
    assert 1 <= len(error_rows) <= 30
    assert error_rows[0].startswith("L97       L97 ")  # anchor label, nesting indent, then the exact line
    assert any("(x" in row for row in error_rows)  # the Traceback line repeats and collapses
    assert "errors" in text


def test_shebang_and_unknown_extension():
    tool, _ = artifact_for("shebang")
    assert structure.detect_language("bin/tool", "#!/usr/bin/env python\n") == "python"
    assert structure.outline_for(tool).parser == "ast"
    other, _ = artifact_for("unknown.ext")
    assert "symbols" not in applicable_rungs(other)


# --- P2 / P5 / P11: a view never costs more than the rung above --------------------------------

def _byte_meter(text: str):
    data = text.encode("utf-8", errors="surrogatepass")
    return len(data), (len(data) + 3) // 4


def test_view_never_costs_more_than_rung_above():
    rng = random.Random(1234)
    sources = [name for name in FIXTURES if name not in ("empty.py", "binary.bin")]
    for _ in range(60):
        name = rng.choice(sources)
        full_art, text = artifact_for(name)
        total = full_art.total_lines
        start = rng.randint(1, total)
        end = rng.randint(start, min(total, start + rng.randint(0, 80)))
        artifact, _ = artifact_for(name, start, end)
        renders = _renders(artifact, _page_text(artifact))
        candidates = candidate_rungs(renders, current="full", meter=_byte_meter)
        costs = [_byte_meter(renders[r]) for r in ["full"] + candidates]
        for above, below in zip(costs, costs[1:]):
            assert below[0] < above[0] and below[1] < above[1], (name, start, end, candidates)
        for rung in set(applicable_rungs(artifact)) - set(candidates):
            assert _byte_meter(renders[rung]) >= _byte_meter(renders["full"]) or any(
                _byte_meter(renders[rung])[i] >= _byte_meter(renders[c])[i] for c in candidates for i in (0, 1))


def test_one_line_file_stays_full_when_every_view_is_larger():
    artifact, _ = artifact_for("one_line.py")
    renders = _renders(artifact, _page_text(artifact))
    # A one-line page is shorter than any degraded carrier (the stub alone names the path twice).
    assert candidate_rungs(renders, current="full", meter=_byte_meter) == []
    assert pick_rung(renders, current="full", deficit_bytes=1, deficit_tokens=1, meter=_byte_meter) is None


def test_choose_rung_requires_token_and_byte_saving():
    renders = {"full": "f" * 100, "symbols": "s" * 60, "stub": "t" * 20}
    assert candidate_rungs(renders, current="full", meter=_byte_meter) == ["symbols", "stub"]

    def token_heavy(text: str):  # symbols is smaller in bytes but denser in tokens
        return len(text), 1000 if text.startswith("s") else len(text) // 4
    assert candidate_rungs(renders, current="full", meter=token_heavy) == ["stub"]
    assert candidate_rungs(renders, current="symbols", meter=_byte_meter) == ["stub"]
    assert candidate_rungs(renders, current="stub", meter=_byte_meter) == []
    assert candidate_rungs(renders, current="full", meter=_byte_meter, keep_floor="symbols") == ["symbols"]
    assert pick_rung(renders, current="full", deficit_bytes=30, deficit_tokens=5, meter=_byte_meter) == "symbols"
    assert pick_rung(renders, current="full", deficit_bytes=70, deficit_tokens=5, meter=_byte_meter) == "stub"
    assert pick_rung(renders, current="full", deficit_bytes=500, deficit_tokens=5, meter=_byte_meter) == "stub"
    with pytest.raises(ValueError):
        candidate_rungs({"stub": "x"}, current="full")


# --- P4: overlapping artifacts render independently -----------------------------------------------

def test_overlapping_artifacts_render_independently():
    first, _ = artifact_for("calc.py", 1, 20)
    second, _ = artifact_for("calc.py", 15, 29)
    a = render_view(first, "symbols", reason="context_limit")
    b = render_view(second, "symbols", reason="context_limit")
    assert a.vid != b.vid
    assert "lines=1-20/29" in a.text and "lines=15-29/29" in b.text
    assert "L23-" in b.text and "def mean" in b.text and "L23-24" not in a.text
    assert "parser=scan" in b.text  # 15-29 mixes indented methods and top-level defs: dedent cannot help
    assert "L15-17    def add" in a.text
    assert 'symbol="add"' in b.text  # the slice alone parses dedented: no class context without the file
    with_file = render_view(second, "symbols", reason="context_limit", file_text=artifact_for("calc.py")[1]).text
    assert "(enclosing, starts L8)" in with_file and 'symbol="Calc.add"' in with_file


def test_relevant_subranges_disjoint():
    artifact, _ = artifact_for("calc.py")
    rendered = render_view(artifact, "relevant", reason="context_limit",
                           frozen_selection=[(1, 10, "unit", "resolved"), (8, 14, "frame", "observed"), (13, 16, "lit", "resolved")])
    rows = list(iter_verbatim_rows(rendered.text))
    assert [(s, e) for s, e, _ in rows] == [(1, 16)]
    assert "--- lines 1-16 (frame, observed+lit, resolved+unit, resolved) ---" in rendered.text
    for start, end, text in rows:
        assert row_matches_slice(text, artifact.slice(start, end))


# --- P10: change notice is never edit authority and never empty ---------------------------------

def test_change_notice_never_edit_authority():
    artifact, text = artifact_for("calc.py")
    new_text = text.replace("DEFAULT_SCALE = 2", "DEFAULT_SCALE = 3")
    new_rev = revision_of(new_text.encode())
    for rung, kwargs in (("stub", {}), ("symbols", {}), ("range", {"ranges": [(5, 5)]})):
        rendered = render_view(artifact, rung, reason="context_limit", disk_revision=new_rev, disk_text=new_text, **kwargs)
        body = rendered.text.split("\n")[1:]
        notice = [line for line in body if line.startswith("changed: ")]
        assert len(notice) == 1 and notice[0] == body[-2]
        assert "DEFAULT_SCALE = 3" not in rendered.text  # no hunk bodies
        assert "@@" not in rendered.text and "\n+" not in rendered.text and "\n-D" not in rendered.text
        assert "degraded=true" in rendered.text  # still a degraded view, attached to real content


def test_empty_change_not_rendered():
    artifact, text = artifact_for("calc.py")
    assert change_notice(artifact, disk_revision=artifact.revision, disk_text=text) is None
    assert change_notice(artifact, disk_revision=None, disk_text=text) is None
    same = render_view(artifact, "stub", reason="context_limit", disk_revision=artifact.revision, disk_text=text)
    plain = render_view(artifact, "stub", reason="context_limit")
    assert same.text == plain.text
    assert "changed:" not in plain.text


def test_change_notice_is_deterministic_for_a_revision_pair():
    artifact, text = artifact_for("calc.py", 8, 20)
    new_text = text.replace("        self.scale = scale\n", "")
    rev = revision_of(new_text.encode())
    a = change_notice(artifact, disk_revision=rev, disk_text=new_text)
    b = change_notice(artifact, disk_revision=rev, disk_text=new_text)
    assert a == b and "held 8-20 maps to 8-19 (+0 -1 lines inside range)" in a


# --- edge cases ------------------------------------------------------------------------------------

def test_empty_file_only_stub():
    artifact, _ = artifact_for("empty.py")
    assert artifact.is_empty and applicable_rungs(artifact) == ("stub",)
    rendered = render_view(artifact, "stub", reason="context_limit")
    assert "lines=1-0/0" in rendered.text
    with pytest.raises(ValueError):
        render_view(artifact, "range", reason="context_limit", ranges=[(1, 1)])


def test_huge_log_renders_bounded_map():
    log, _ = artifact_for("worker.log")
    rendered = render_view(log, "symbols", reason="context_limit")
    assert len(rendered.text.encode()) < 8000 < len(log.held_text)
    rows = list(iter_verbatim_rows(render_view(log, "range", reason="model_keep", ranges=[(2990, 3000)]).text))
    assert rows[0][:2] == (2990, 3000) and row_matches_slice(rows[0][2], log.slice(2990, 3000))


def test_form_feed_line_numbers_agree_with_splitlines():
    artifact, text = artifact_for("form_feed.py")
    assert artifact.total_lines == len(text.splitlines()) == 6  # "\x0c" splits a line, like deliver_source
    rendered = render_view(artifact, "range", reason="model_keep", ranges=[(3, 4)])
    rows = list(iter_verbatim_rows(rendered.text))
    assert rows == [(3, 4, "\x0c\n")] or row_matches_slice(rows[0][2], artifact.slice(3, 4))
    sym = render_view(artifact, "symbols", reason="context_limit").text
    assert "L1-2    def f():" in sym and "L5-6    def g():" in sym


def test_artifact_validation_and_slices():
    rev = revision_of(b"a\nb\nc\n")
    artifact = Artifact(display="x.txt", revision=rev, held_start=2, held_end=3, total_lines=3, held_text="b\nc\n")
    assert artifact.line(2) == "b\n" and artifact.slice(2, 3) == "b\nc\n"
    with pytest.raises(IndexError):
        artifact.line(1)
    with pytest.raises(ValueError):
        Artifact(display="x", revision="sha256:short", held_start=1, held_end=1, total_lines=1, held_text="a\n")
    with pytest.raises(ValueError):
        Artifact(display="x", revision=rev, held_start=1, held_end=3, total_lines=2, held_text="a\nb\nc\n")
