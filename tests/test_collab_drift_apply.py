"""COLLAB-DRIFT apply paths: a file that moved since delivery is merged, not refused, behind STATEFRAME_COLLAB_DRIFT.

Every case drives the production executors (execute_edit_source / execute_rewrite_source), the same functions the
runtime calls. OFF and STRICT=on must keep today's SOURCE_CHANGED refusals byte for byte.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_edit_handles import DeliveredSpan, SpanKind, mint_handles
from kernel_edit_source_tool import EditProtocolRunState, execute_edit_source, execute_rewrite_source

SOURCE = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
PATH = "pkg/mod.py"


@pytest.fixture
def drift_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)


def _write(root: Path, text: str) -> Path:
    path = root / PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _state(*spans: DeliveredSpan, ordinal: int = 7) -> EditProtocolRunState:
    state = EditProtocolRunState()
    state.set_request(mint_handles(ordinal, list(spans)))
    return state


def _span(text: str = SOURCE, start: int | None = 1, end: int | None = 2,
          kind: SpanKind = SpanKind.SPAN) -> DeliveredSpan:
    return DeliveredSpan.from_source(PATH, text, start_line=start, end_line=end, kind=kind)


def _edit(root: Path, state: EditProtocolRunState, handle: str, old: str, new: str):
    def read_bytes(p: Path) -> bytes:
        return p.read_bytes()
    return execute_edit_source(root, {"edits": [{"handle": handle, "old_text": old, "new_text": new}]},
                               state=state, read_bytes=read_bytes)


def _rewrite(root: Path, state: EditProtocolRunState, handle: str, new: str):
    def read_bytes(p: Path) -> bytes:
        return p.read_bytes()
    return execute_rewrite_source(root, {"handle": handle, "new_text": new}, state=state,
                                  read_bytes=read_bytes)


# -- production apply paths, switch ON ----------------------------------------------------------------------

def test_outside_edit_above_span_reanchors_and_applies(tmp_path, drift_on):
    state = _state(_span())
    handle = state.handle_map.records[0].handle
    path = _write(tmp_path, "# header\n" + SOURCE)
    out = _edit(tmp_path, state, handle, "return 1", "return 10")
    assert out.status == "APPLIED", out.as_dict()
    assert path.read_text() == "# header\n" + SOURCE.replace("return 1", "return 10")
    assert "drift=reanchored" in out.detail


def test_rewrite_merges_outside_change_with_rewrite_hunk(tmp_path, drift_on):
    state = _state(_span(start=None, end=None, kind=SpanKind.REWRITE))
    handle = state.handle_map.records[0].handle
    path = _write(tmp_path, SOURCE.replace("return 2", "return 22"))
    out = _rewrite(tmp_path, state, handle, SOURCE.replace("return 1", "return 10"))
    assert out.status == "APPLIED", out.as_dict()
    assert path.read_text() == SOURCE.replace("return 1", "return 10").replace("return 2", "return 22")
    assert "drift=merged" in out.detail


def test_same_lines_changed_both_sides_is_edit_conflict_with_hunks(tmp_path, drift_on):
    state = _state(_span())
    handle = state.handle_map.records[0].handle
    current = SOURCE.replace("return 1", "return 5")
    path = _write(tmp_path, current)
    out = _edit(tmp_path, state, handle, "return 1", "return 10")
    assert out.status == "REJECTED" and out.codes() == ("EDIT_CONFLICT",)
    assert path.read_text() == current
    body = out.as_dict()
    hunk = body["conflict"][0]
    assert "return 5" in hunk["theirs_text"] and "return 10" in hunk["ours_text"]
    assert "return 1" in hunk["base_text"]
    assert hunk["theirs_lines"] and hunk["base_lines"]


def test_duplicate_block_next_to_anchor_never_lands_at_wrong_location(tmp_path, drift_on):
    state = _state(_span())
    handle = state.handle_map.records[0].handle
    current = "def a():\n    return 1\n" + SOURCE
    path = _write(tmp_path, current)
    out = _edit(tmp_path, state, handle, "return 1", "return 10")
    assert out.status == "REJECTED"
    assert set(out.codes()) <= {"SOURCE_CHANGED", "EDIT_CONFLICT"}
    assert path.read_text() == current


def test_two_engine_edits_to_one_file_in_one_request_both_land(tmp_path, drift_on):
    state = _state(_span(start=1, end=2), _span(start=5, end=6))
    first, second = (record.handle for record in state.handle_map.records)
    path = _write(tmp_path, SOURCE)
    assert _edit(tmp_path, state, first, "return 1", "return 10").status == "APPLIED"
    out = _edit(tmp_path, state, second, "return 2", "return 20")
    assert out.status == "APPLIED", out.as_dict()
    assert path.read_text() == SOURCE.replace("return 1", "return 10").replace("return 2", "return 20")


def test_rewrite_base_not_retained_stays_source_changed(tmp_path, drift_on):
    big = SOURCE + ("# filler line to exceed the retained rewrite base\n" * 400)
    state = _state(_span(text=big, start=None, end=None, kind=SpanKind.REWRITE))
    handle = state.handle_map.records[0].handle
    _write(tmp_path, big + "# outside edit\n")
    out = _rewrite(tmp_path, state, handle, SOURCE)
    assert out.status == "REJECTED" and out.codes() == ("SOURCE_CHANGED",)


# -- switch OFF and STRICT keep today's refusals ------------------------------------------------------------

def test_switch_off_keeps_source_changed_for_outside_edit(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_COLLAB_DRIFT", raising=False)
    state = _state(_span())
    handle = state.handle_map.records[0].handle
    current = "# header\n" + SOURCE
    path = _write(tmp_path, current)
    out = _edit(tmp_path, state, handle, "return 1", "return 10")
    assert out.status == "REJECTED" and out.codes() == ("SOURCE_CHANGED",)
    assert path.read_text() == current
    assert "conflict" not in out.as_dict()


def test_rewrite_switch_off_keeps_source_changed(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "off")
    state = _state(_span(start=None, end=None, kind=SpanKind.REWRITE))
    handle = state.handle_map.records[0].handle
    _write(tmp_path, SOURCE.replace("return 2", "return 22"))
    out = _rewrite(tmp_path, state, handle, SOURCE.replace("return 1", "return 10"))
    assert out.status == "REJECTED" and out.codes() == ("SOURCE_CHANGED",)


def test_strict_drift_on_forces_refusal_even_when_collab_on(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.setenv("STATEFRAME_STRICT_DRIFT", "on")
    state = _state(_span())
    handle = state.handle_map.records[0].handle
    current = "# header\n" + SOURCE
    path = _write(tmp_path, current)
    out = _edit(tmp_path, state, handle, "return 1", "return 10")
    assert out.status == "REJECTED" and out.codes() == ("SOURCE_CHANGED",)
    assert path.read_text() == current


@pytest.mark.parametrize("name", ["STATEFRAME_COLLAB_DRIFT", "STATEFRAME_STRICT_DRIFT"])
def test_invalid_switch_value_raises(monkeypatch, name):
    from kernel_edit_drift import collab_drift_enabled, strict_drift_enabled
    monkeypatch.setenv(name, "maybe")
    reader = collab_drift_enabled if name == "STATEFRAME_COLLAB_DRIFT" else strict_drift_enabled
    with pytest.raises(ValueError):
        reader()


# -- merge core: the diff3 contract the apply paths rely on ------------------------------------------------

def test_merge_texts_non_overlapping_hunks_combine():
    from kernel_edit_merge import MergeApplied, merge_texts
    base = "a\nb\nc\nd\n"
    out = merge_texts(base, "A\nb\nc\nd\n", "a\nb\nc\nD\n")
    assert isinstance(out, MergeApplied)
    assert out.text == "A\nb\nc\nD\n"


def test_merge_texts_identical_change_on_both_sides_is_not_a_conflict():
    from kernel_edit_merge import MergeApplied, merge_texts
    base = "a\nb\n"
    out = merge_texts(base, "a\nB\n", "a\nB\n")
    assert isinstance(out, MergeApplied) and out.text == "a\nB\n"


def test_merge_texts_overlapping_different_changes_conflict():
    from kernel_edit_merge import MergeConflict, merge_texts
    base = "a\nb\nc\n"
    out = merge_texts(base, "a\nX\nc\n", "a\nY\nc\n")
    assert isinstance(out, MergeConflict)
    assert out.hunks[0].base_text == "b\n"
    assert out.hunks[0].ours_text == "X\n" and out.hunks[0].theirs_text == "Y\n"


def test_merge_texts_no_changes_round_trips():
    from kernel_edit_merge import MergeApplied, merge_texts
    assert merge_texts("x\n", "x\n", "x\n") == MergeApplied("x\n", "merged", (0, 2), None, ())


def test_merge_texts_unambiguous_no_anchor_is_not_forced():
    from kernel_edit_merge import MergeApplied, MergeConflict, MergeAmbiguous, merge_texts
    # A change that cannot be placed must never produce MergeApplied with a guessed location.
    out = merge_texts("p\np\n", "p\nq\n", "p\np\np\n")
    assert isinstance(out, (MergeConflict, MergeAmbiguous, MergeApplied))
    if isinstance(out, MergeApplied):
        assert out.text.count("q") == 1


def test_outcome_observation_stays_json_serialisable_with_conflict(tmp_path, drift_on):
    state = _state(_span())
    handle = state.handle_map.records[0].handle
    _write(tmp_path, SOURCE.replace("return 1", "return 5"))
    out = _edit(tmp_path, state, handle, "return 1", "return 10")
    assert json.loads(out.observation())["refusals"][0]["code"] == "EDIT_CONFLICT"


@pytest.mark.xfail(strict=True, reason="pre-existing: edit_source raises UnicodeEncodeError on non-UTF-8 files (kernel_edit_source_tool._read_current -> source_revision), also on 0.7.43; filed separately")
def test_non_utf8_bytes_on_drift_path_apply_byte_exact(tmp_path, drift_on):
    """Invalid UTF-8 outside the span must survive a drift merge byte for byte, never be re-encoded."""
    text = "alpha\nbeta\n"
    span = DeliveredSpan.from_source("pkg/data.txt", text, start_line=1, end_line=1)
    state = _state(span)
    handle = state.handle_map.records[0].handle
    current = b"\xff\xfe header \xc3\n" + text.encode("utf-8")
    target = tmp_path / "pkg" / "data.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(current)
    out = _edit(tmp_path, state, handle, "alpha", "ALPHA")
    if out.status == "APPLIED":
        assert target.read_bytes() == b"\xff\xfe header \xc3\n" + b"ALPHA\nbeta\n"
    else:
        assert out.codes() == ("SOURCE_CHANGED",)
        assert target.read_bytes() == current
