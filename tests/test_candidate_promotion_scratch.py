"""Regression coverage for scratch omitted from candidate promotion digests."""
from __future__ import annotations

import hashlib
from pathlib import Path

from stateframe import runner, workcopy


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_witness_summary_ignores_only_omitted_scratch() -> None:
    scratch_path = ".test-tmp/from-prior-pytest-run.txt"
    baseline = {
        "selection.py": _digest(b"before\n"),
        scratch_path: _digest(b"scratch\n"),
    }
    candidate = {"selection.py": _digest(b"after\n")}

    ordinary = runner._promotion_changes_for_summary(baseline, candidate)
    witness = runner._promotion_changes_for_summary(
        baseline, candidate, schedule_witness_enabled=True)
    witness_with_scratch = runner._promotion_changes_for_summary(
        baseline, {**candidate, scratch_path: baseline[scratch_path]},
        schedule_witness_enabled=True)

    assert ordinary == {".test-tmp/from-prior-pytest-run.txt": "deleted",
                       "selection.py": "modified"}
    assert witness == {"selection.py": "modified"}
    assert witness_with_scratch == {"selection.py": "modified"}


def test_in_place_preserves_scratch_omitted_from_witness_digest(tmp_path: Path) -> None:
    original = tmp_path / "repo"
    candidate = tmp_path / "candidate"
    original.mkdir()
    candidate.mkdir()
    original_selection = b"def choose(): return 'old'\n"
    candidate_selection = b"def choose(): return 'new'\n"
    scratch_bytes = b"scratch from a prior local pytest run\n"
    (original / "selection.py").write_bytes(original_selection)
    scratch = original / ".test-tmp" / "from-prior-pytest-run.txt"
    scratch.parent.mkdir()
    scratch.write_bytes(scratch_bytes)
    (candidate / "selection.py").write_bytes(candidate_selection)

    baseline = {
        "selection.py": _digest(original_selection),
        ".test-tmp/from-prior-pytest-run.txt": _digest(scratch_bytes),
    }
    candidate_digests = {"selection.py": _digest(candidate_selection)}
    changed = runner._promotion_changes_for_summary(
        baseline, candidate_digests, schedule_witness_enabled=True)
    work = workcopy.WorkCopy(original=original, copy=candidate, baseline=baseline)

    applied = workcopy.apply_in_place(
        work, changed=changed, candidate_file_digests=candidate_digests)

    assert applied == ["selection.py"]
    assert (original / "selection.py").read_bytes() == candidate_selection
    assert scratch.read_bytes() == scratch_bytes
