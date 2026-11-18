"""Sealing must preserve workspace bytes before checking the delivery scope."""

import json
from pathlib import Path

import pytest
from kernel_done_is_submit_authority import store_candidate
from probes import react_repair

from stateframe import runner, workcopy


@pytest.mark.parametrize("stat_snapshot", [False, True], ids=["direct", "stat"])
@pytest.mark.parametrize("raw", [
    b"\x03\x00\x00\x00\r\x00record\r\n\x1a",
    b"value = 1\r\n",
    b"a\rb\r\nc\n",
    b"\xff\r\x00",
    b"plain\n",
    "caf\u00e9\r\n".encode("utf-8"),
    b"",
], ids=["utf8-binary", "crlf", "mixed-newlines", "non-utf8", "lf", "unicode", "empty"])
def test_sealed_candidate_preserves_bytes_and_delivers_real_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes, stat_snapshot: bool,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    fixtures = {
        "fixture.dbf": raw,
        "legacy.dbf": b"\x03\x00\xff\x80\r\nrecord\r\x00\x1a",
    }
    for path, data in fixtures.items():
        (root / path).write_bytes(data)
    (root / "app.py").write_bytes(b"value = 1\n")
    (root / "edited.dbf").write_bytes(b"original\x00\xff")
    (root / "removed.py").write_bytes(b"obsolete = True\n")
    work = workcopy.create(root, tmp_path / "delivery")

    # Only these declared files are model edits; neither fixture is editable.
    declared = {"app.py", "edited.dbf", "removed.py"}
    (root / "app.py").write_bytes(b"value = 2\n")
    (root / "edited.dbf").write_bytes(raw)
    (root / "removed.py").unlink()
    expected = {**fixtures, "app.py": b"value = 2\n", "edited.dbf": raw, "removed.py": None}
    monkeypatch.setattr(react_repair, "_SNAPSHOT_STAT_LIMIT", 0 if stat_snapshot else 100)
    snapshot = react_repair._source_snapshot(root, tuple(expected))
    sources = react_repair._seal_sources_preserving_binary(root, snapshot)

    run = tmp_path / "run"
    stored = store_candidate(run / "worker/results/react_done_submissions", "task", sources)
    digest = stored["artifact_digest"]
    (run / "engine-result.json").write_text(json.dumps({
        "result": {"result": "SUBMITTED", "submission": {
            "stored": True, "seal_id": "sha256:" + "0" * 64,
            "artifact_digest": digest}}}), encoding="utf-8")
    candidate = runner.final_candidate(run)
    assert candidate is not None
    assert runner._dropped_existing_candidate_paths(work, candidate, declared) == []
    assert candidate == expected
    assert runner._artifact_digest(candidate) == digest
    assert all((root / path).read_bytes() == data for path, data in fixtures.items())

    # An actual fixture mutation must still be rejected by the same scope boundary.
    tampered = {**candidate, "fixture.dbf": raw + b"changed"}
    assert runner._dropped_existing_candidate_paths(work, tampered, declared) == ["fixture.dbf"]

    assert workcopy.apply_files(work, candidate, declared=declared) == sorted(declared)
    assert workcopy.changed_files(work) == {
        "app.py": "modified", "edited.dbf": "modified", "removed.py": "deleted",
    }
    for path, data in expected.items():
        target = work.copy / path
        if data is None:
            assert not target.exists()
        else:
            assert target.read_bytes() == data
