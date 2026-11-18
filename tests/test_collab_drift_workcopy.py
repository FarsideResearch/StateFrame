"""COLLAB-DRIFT piece F: --in-place merges outside edits instead of refusing, behind STATEFRAME_COLLAB_DRIFT.

Every case drives the production --in-place entry point (workcopy.apply_in_place) on a working copy set up the way
runner.prepare sets it up (copy, start snapshot in run/parent, snapshot wired on the WorkCopy). OFF, STRICT=on and
any value other than "on" must keep today's refusal text byte for byte.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from stateframe import workcopy

SOURCE = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
MOD = "pkg/mod.py"
REFUSED = "changed in the original since the start"


@pytest.fixture
def drift_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)


def _start(tmp_path: Path, files: dict[str, bytes | str]) -> workcopy.WorkCopy:
    original = tmp_path / "repo"
    for rel, data in files.items():
        path = original / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    run = tmp_path / "run"
    run.mkdir()
    work = workcopy.create(original, run / "workspace")
    shutil.copytree(work.copy, run / "parent", symlinks=False)
    work.snapshot = run / "parent"
    return work


def _run_writes(work: workcopy.WorkCopy, rel: str, text: str) -> None:
    (work.copy / rel).write_bytes(text.encode("utf-8"))


def _human_writes(work: workcopy.WorkCopy, rel: str, data: bytes, *, keep_mtime: bool = False) -> None:
    path = work.original / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    before = path.stat() if path.exists() else None
    path.write_bytes(data)
    if keep_mtime and before is not None:
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _original_bytes(work: workcopy.WorkCopy, rel: str) -> bytes | None:
    path = work.original / rel
    return path.read_bytes() if path.is_file() else None


# -- ON: outside edits merge ---------------------------------------------------------------------------------


def test_different_lines_merge_and_both_changes_land(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE, "other.txt": "x\n"})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    _human_writes(work, MOD, SOURCE.replace("return 2", "return 20").encode("utf-8"))

    merged: list[str] = []
    applied = workcopy.apply_in_place(work, merged=merged)

    assert _original_bytes(work, MOD) == SOURCE.replace("return 1", "return 10").replace(
        "return 2", "return 20").encode("utf-8")
    assert applied == [MOD]
    assert merged == [MOD]


def test_overlapping_lines_conflict_nothing_written_and_hunks_reported(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE, "notes.txt": "a\n"})
    run_version = SOURCE.replace("return 1", "return 10")
    human_version = SOURCE.replace("return 1", "return 11")
    _run_writes(work, MOD, run_version)
    _run_writes(work, "notes.txt", "b\n")
    _human_writes(work, MOD, human_version.encode("utf-8"))

    with pytest.raises(workcopy.MergeConflictError) as raised:
        workcopy.apply_in_place(work, merged=[])

    message = str(raised.value)
    assert "refusing --in-place" in message
    assert MOD in message and "merge conflict" in message
    assert raised.value.conflicts[0].path == MOD
    assert raised.value.conflicts[0].hunk_count == 1
    hunk = raised.value.conflicts[0].hunks[0]
    assert hunk["base_lines"] == [2, 2]
    assert hunk["run_lines"] == [2, 2]
    assert hunk["original_lines"] == [2, 2]
    # All or nothing: the clean notes.txt change is not written either, and the human's version stays.
    assert _original_bytes(work, MOD) == human_version.encode("utf-8")
    assert _original_bytes(work, "notes.txt") == b"a\n"


def test_same_size_edit_with_preserved_mtime_is_still_merged(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    # Same byte length on another line, and the mtime is restored: only the content can reveal it.
    _human_writes(work, MOD, SOURCE.replace("return 2", "return 9").encode("utf-8"), keep_mtime=True)

    merged: list[str] = []
    workcopy.apply_in_place(work, merged=merged)

    assert _original_bytes(work, MOD) == SOURCE.replace("return 1", "return 10").replace(
        "return 2", "return 9").encode("utf-8")
    assert merged == [MOD]


def test_crlf_line_endings_survive_a_merge(tmp_path, drift_on):
    crlf = b"alpha\r\nbeta\r\ngamma\r\n"
    work = _start(tmp_path, {"lines.txt": crlf})
    (work.copy / "lines.txt").write_bytes(b"ALPHA\r\nbeta\r\ngamma\r\n")
    _human_writes(work, "lines.txt", b"alpha\r\nbeta\r\nGAMMA\r\n")

    workcopy.apply_in_place(work, merged=[])

    assert _original_bytes(work, "lines.txt") == b"ALPHA\r\nbeta\r\nGAMMA\r\n"


def test_unchanged_by_run_file_edited_by_human_is_left_alone(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE, "other.txt": "x\n"})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    _human_writes(work, "other.txt", b"human\n")

    merged: list[str] = []
    applied = workcopy.apply_in_place(work, merged=merged)

    assert applied == [MOD]
    assert merged == []
    assert _original_bytes(work, "other.txt") == b"human\n"


# -- ON: cases that stay refusals ----------------------------------------------------------------------------


def test_human_deleted_file_is_refused_as_today(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    (work.original / MOD).unlink()

    with pytest.raises(workcopy.WorkCopyError, match=REFUSED):
        workcopy.apply_in_place(work)
    assert _original_bytes(work, MOD) is None


def test_human_created_same_path_is_refused_as_today(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, "pkg/new.py", "print(1)\n")
    _human_writes(work, "pkg/new.py", b"print(2)\n")

    with pytest.raises(workcopy.WorkCopyError, match="was created in the original meanwhile"):
        workcopy.apply_in_place(work)
    assert _original_bytes(work, "pkg/new.py") == b"print(2)\n"


def test_non_utf8_original_is_refused_byte_safe(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    human = SOURCE.replace("return 2", "return 20").encode("utf-8") + b"# caf\xe9\n"
    _human_writes(work, MOD, human)

    with pytest.raises(workcopy.WorkCopyError, match=REFUSED):
        workcopy.apply_in_place(work)
    assert _original_bytes(work, MOD) == human


def test_nul_byte_file_is_refused_not_merged(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    human = SOURCE.replace("return 2", "return 20").encode("utf-8") + b"\x00\n"
    _human_writes(work, MOD, human)

    with pytest.raises(workcopy.WorkCopyError, match=REFUSED):
        workcopy.apply_in_place(work)
    assert _original_bytes(work, MOD) == human


def test_missing_start_snapshot_keeps_refusal(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    _human_writes(work, MOD, SOURCE.replace("return 2", "return 20").encode("utf-8"))
    work.snapshot = None

    with pytest.raises(workcopy.WorkCopyError, match=REFUSED):
        workcopy.apply_in_place(work)
    assert _original_bytes(work, MOD) == SOURCE.replace("return 2", "return 20").encode("utf-8")


def test_start_snapshot_bytes_that_do_not_match_baseline_keep_refusal(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    _human_writes(work, MOD, SOURCE.replace("return 2", "return 20").encode("utf-8"))
    (work.snapshot / MOD).write_bytes(b"tampered\n")

    with pytest.raises(workcopy.WorkCopyError, match=REFUSED):
        workcopy.apply_in_place(work)


# -- OFF / STRICT / invalid: today's behaviour, byte for byte ----------------------------------------------


def _expected_refusal() -> str:
    return f"refusing --in-place: {MOD} ({REFUSED})"


def _prepare_overlap_free_edit(tmp_path: Path) -> workcopy.WorkCopy:
    work = _start(tmp_path, {MOD: SOURCE})
    _run_writes(work, MOD, SOURCE.replace("return 1", "return 10"))
    _human_writes(work, MOD, SOURCE.replace("return 2", "return 20").encode("utf-8"))
    return work


def test_off_refuses_exactly_as_today(tmp_path, monkeypatch):
    monkeypatch.delenv("STATEFRAME_COLLAB_DRIFT", raising=False)
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)
    work = _prepare_overlap_free_edit(tmp_path)
    human = _original_bytes(work, MOD)

    with pytest.raises(workcopy.WorkCopyError) as raised:
        workcopy.apply_in_place(work)
    assert str(raised.value) == _expected_refusal()
    assert _original_bytes(work, MOD) == human


def test_strict_on_forces_refusal_even_when_collab_on(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.setenv("STATEFRAME_STRICT_DRIFT", "on")
    work = _prepare_overlap_free_edit(tmp_path)

    with pytest.raises(workcopy.WorkCopyError) as raised:
        workcopy.apply_in_place(work)
    assert str(raised.value) == _expected_refusal()


def test_unrecognised_switch_value_behaves_as_off(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "yes-please")
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)
    work = _prepare_overlap_free_edit(tmp_path)

    with pytest.raises(workcopy.WorkCopyError) as raised:
        workcopy.apply_in_place(work)
    assert str(raised.value) == _expected_refusal()


# -- candidate apply (copy-side writes) is not affected by the drift switch ---------------------------------


def test_candidate_apply_writes_the_copy_with_switch_on(tmp_path, drift_on):
    work = _start(tmp_path, {MOD: SOURCE})
    changed = SOURCE.replace("return 1", "return 10").encode("utf-8")

    written = workcopy.apply_files(work, {MOD: changed}, declared={MOD})

    assert written == [MOD]
    assert (work.copy / MOD).read_bytes() == changed
    assert _original_bytes(work, MOD) == SOURCE.encode("utf-8")
