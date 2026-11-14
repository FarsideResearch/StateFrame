"""View ladder determinism (spec 3.3): same input -> same bytes, exact slices, state round trip."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from _view_ladder_fixtures import FIXTURES, artifact_for
from kernel_view_ladder import (
    ArtifactKey,
    ArtifactState,
    ViewLadderState,
    applicable_rungs,
    iter_verbatim_rows,
    parse_view_header,
    render_view,
    row_matches_slice,
    verify_body_sha256,
)
from kernel_view_ladder.render import DEGRADED_RUNGS

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
TESTS = Path(__file__).resolve().parent

_RENDER_ALL = r"""
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from _view_ladder_fixtures import FIXTURES, artifact_for
from kernel_view_ladder import applicable_rungs, render_view
out = {}
for name in sorted(FIXTURES):
    artifact, _ = artifact_for(name)
    for rung in applicable_rungs(artifact):
        kwargs = {}
        if rung == "range":
            kwargs["ranges"] = [(artifact.held_start, min(artifact.held_end, artifact.held_start + 1))]
        if rung == "relevant":
            kwargs["frozen_selection"] = [(artifact.held_start, artifact.held_start, "frame", "observed")]
        r = render_view(artifact, rung, reason="context_limit", **kwargs)
        out[f"{name}:{rung}"] = [r.vid, r.body_sha256, r.text]
print(json.dumps(out, sort_keys=True))
"""


def _render_in_subprocess() -> dict[str, list[str]]:
    result = subprocess.run([sys.executable, "-c", _RENDER_ALL, str(TESTS), str(ENGINE)],
                            capture_output=True, text=True, check=True, timeout=120)
    return json.loads(result.stdout)


def test_two_fresh_processes_render_identical_bytes():
    first, second = _render_in_subprocess(), _render_in_subprocess()
    assert first == second
    assert len(first) >= 25 * 3  # every fixture renders at least range/relevant/stub (or stub alone when empty)
    for key, (vid, body_sha, text) in first.items():
        header = parse_view_header(text)
        assert header and header["vid"] == vid and header["body_sha256"] == body_sha, key
        assert verify_body_sha256(text), key


def _rows_kwargs(artifact, rung):
    if rung == "range":
        return {"ranges": [(artifact.held_start, artifact.held_end), (artifact.held_start, artifact.held_start)]}
    if rung == "relevant":
        mid = (artifact.held_start + artifact.held_end) // 2
        return {"frozen_selection": [(mid, artifact.held_end, "unit", "resolved"), (artifact.held_start, mid, "frame", "observed")]}
    return {}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_verbatim_rows_are_exact_slices_of_the_held_text(name):
    artifact, _ = artifact_for(name)
    for rung in applicable_rungs(artifact):
        rendered = render_view(artifact, rung, reason="context_limit", **_rows_kwargs(artifact, rung))
        if rung in ("range", "relevant"):
            rows = list(iter_verbatim_rows(rendered.text))
            assert rows, name
            for start, end, text in rows:
                assert row_matches_slice(text, artifact.slice(start, end)), (name, start, end)
        # structure rows are checked against the held lines in test_view_ladder_pitfalls.py



def test_crlf_bom_and_replaced_bytes_survive_verbatim_rows():
    for name in ("crlf.py", "bom.py", "invalid_utf8.py", "mixed_endings.py", "form_feed.py", "no_trailing_newline.py"):
        artifact, _ = artifact_for(name)
        rendered = render_view(artifact, "range", reason="model_keep", ranges=[(artifact.held_start, artifact.held_end)])
        rows = list(iter_verbatim_rows(rendered.text))
        assert len(rows) == 1
        assert row_matches_slice(rows[0][2], artifact.held_text)
    crlf, _ = artifact_for("crlf.py")
    assert "\r\n" in render_view(crlf, "range", reason="model_keep", ranges=[(2, 3)]).text
    assert "�" in render_view(artifact_for("invalid_utf8.py")[0], "range", reason="model_keep", ranges=[(2, 2)]).text
    header = render_view(crlf, "range", reason="model_keep", ranges=[(2, 3)]).text.split("\n", 1)[0]
    assert "\r" not in header


def test_mixed_line_endings_are_recorded_not_fixed():
    artifact, _ = artifact_for("mixed_endings.py")
    rendered = render_view(artifact, "symbols", reason="context_limit")
    assert "line_endings=mixed" in rendered.text


def test_input_order_of_ranges_does_not_change_bytes():
    artifact, _ = artifact_for("calc.py")
    a = render_view(artifact, "relevant", reason="context_limit",
                    frozen_selection=[(24, 25, "frame", "observed"), (15, 20, "unit", "resolved")])
    b = render_view(artifact, "relevant", reason="context_limit",
                    frozen_selection=[(15, 20, "unit", "resolved"), (24, 25, "frame", "observed")])
    assert a.text == b.text and a.vid == b.vid


def test_vid_changes_with_selection_rung_and_revision():
    artifact, _ = artifact_for("calc.py")
    base = render_view(artifact, "range", reason="model_keep", ranges=[(15, 20)])
    other = render_view(artifact, "range", reason="model_keep", ranges=[(15, 21)])
    assert base.vid != other.vid
    assert base.vid != render_view(artifact, "stub", reason="model_keep").vid
    moved, _ = artifact_for("crlf.py")
    assert render_view(moved, "stub", reason="model_keep").vid != render_view(artifact, "stub", reason="model_keep").vid


def test_state_json_round_trip_re_renders_identically():
    artifact, _ = artifact_for("calc.py")
    key = ArtifactKey(artifact.display, artifact.revision, artifact.held_start, artifact.held_end)
    state = ViewLadderState()
    entry = ArtifactState(held_text=artifact.held_text, fresh_turn=3)
    entry.frozen_selection = ((15, 20, "unit", "resolved"), (24, 25, "frame", "observed"))
    rendered = render_view(artifact, "relevant", reason="context_limit", frozen_selection=entry.frozen_selection)
    entry.record_render("relevant", rendered.text)
    entry.set_rung("relevant", "context_limit")
    state.stage(key, entry)
    state.commit()
    blob = state.to_json()
    restored = ViewLadderState.from_json(blob)
    back = restored.entries[key]
    assert back.rendered["relevant"] == entry.rendered["relevant"]
    assert back.frozen_selection == entry.frozen_selection
    again = render_view(artifact, "relevant", reason="context_limit", frozen_selection=back.frozen_selection)
    assert again.text == rendered.text and again.vid == rendered.vid == back.vid(key)
    assert restored.to_json() == blob


def test_record_render_is_immutable():
    entry = ArtifactState()
    entry.record_render("stub", "a")
    entry.record_render("stub", "a")
    with pytest.raises(ValueError):
        entry.record_render("stub", "b")


def test_pending_commit_and_rollback():
    key = ArtifactKey("x.py", "sha256:" + "0" * 64, 1, 2)
    state = ViewLadderState()
    state.stage(key, ArtifactState(rung="stub"))
    assert state.get(key).rung == "stub" and key not in state.entries
    state.rollback()
    assert state.get(key) is None
    state.stage(key, ArtifactState(rung="symbols"))
    state.commit()
    assert state.entries[key].rung == "symbols" and not state.pending


def test_every_degraded_rung_renders_for_every_fixture_where_applicable():
    for name in FIXTURES:
        artifact, _ = artifact_for(name)
        rungs = applicable_rungs(artifact)
        assert "stub" in rungs
        for rung in DEGRADED_RUNGS:
            kwargs = _rows_kwargs(artifact, rung)
            if rung in rungs:
                render_view(artifact, rung, reason="context_limit", **kwargs)
            elif rung in ("symbols", "outline"):
                with pytest.raises(ValueError):
                    render_view(artifact, rung, reason="context_limit")
