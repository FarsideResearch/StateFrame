"""Outcome receipt recovery from durable subframe completion evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe import outcomes
from stateframe.engine.kernel_parallel_records import CheckReceipt


def _write_auto_finished_submission(run: Path, *, artifact_digest: str,
                                    seal_id: str) -> dict[str, object]:
    journal = run / "worker" / "measurement-runs" / "one" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    receipt = CheckReceipt(
        "r" * 64, "p" * 64, "h" * 64, "e" * 64, "policy",
        "PASS", None, (), str(run / "check-evidence.txt"), "pytest exit 0",
    ).to_json()
    source_map_root = run / "worker" / "results" / "react_final" / "execution"
    source_map_root.mkdir(parents=True)
    source_map_path = source_map_root / "source_map.json"
    source_map_path.write_text(json.dumps({"files": {"pkg/a.py": "VALUE = 2\n"}}),
                               encoding="utf-8")
    events = [
        {"kind": "subframe_auto_finished", "check_receipt": receipt},
        {"kind": "subframe_final_sources",
         "artifact_digest": artifact_digest, "source_map_path": str(source_map_path),
         "final_source_paths": ["pkg/a.py"]},
        {"kind": "done_sealed", "stored": True, "source": "host",
         "seal_id": seal_id, "artifact_digest": artifact_digest},
        {"kind": "done_submission_stored", "stored": True,
         "artifact_digest": artifact_digest},
    ]
    journal.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return receipt


@pytest.mark.parametrize("has_terminal_file", [False, True], ids=["journal-fallback", "terminal-enrichment"])
def test_terminal_receipt_preserves_auto_finish_check_receipt(
        tmp_path: Path, has_terminal_file: bool) -> None:
    run = tmp_path / "run"
    artifact_digest = "sha256:" + "a" * 64
    seal_id = "sha256:" + "b" * 64
    receipt = _write_auto_finished_submission(
        run, artifact_digest=artifact_digest, seal_id=seal_id)
    if has_terminal_file:
        (run / "engine-result.json").write_text(json.dumps({"result": {
            "result": "SUBMITTED",
            "submission": {"stored": True, "seal_id": seal_id,
                           "artifact_digest": artifact_digest},
        }}), encoding="utf-8")

    terminal = outcomes.terminal_receipt(run)

    assert terminal["result"] == "SUBMITTED"
    assert terminal["check_receipt"] == receipt
    assert terminal["source_map_path"] == str(
        run / "worker" / "results" / "react_final" / "execution" / "source_map.json")
    assert terminal["final_source_paths"] == ["pkg/a.py"]


def test_terminal_receipt_does_not_borrow_check_from_another_seal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_auto_finished_submission(
        run, artifact_digest="sha256:" + "a" * 64,
        seal_id="sha256:" + "b" * 64)
    (run / "engine-result.json").write_text(json.dumps({"result": {
        "result": "SUBMITTED",
        "submission": {"stored": True, "seal_id": "sha256:" + "d" * 64,
                       "artifact_digest": "sha256:" + "c" * 64},
    }}), encoding="utf-8")

    terminal = outcomes.terminal_receipt(run)

    assert terminal["result"] == "SUBMITTED"
    assert "check_receipt" not in terminal
    assert "source_map_path" not in terminal
    assert "final_source_paths" not in terminal
