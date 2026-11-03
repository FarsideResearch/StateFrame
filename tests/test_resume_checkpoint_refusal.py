"""Checkpoint selection records missing pooled blobs before falling back to replay."""
from __future__ import annotations

import hashlib
import json

from stateframe import resume


def test_missing_pooled_blob_is_recorded_loudly_and_checkpoint_is_skipped(tmp_path):
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    folder = checkpoints / "0001-1"
    folder.mkdir(parents=True)
    state = b"checkpoint state"
    (folder / "state.pkl").write_bytes(state)
    manifest = json.dumps({"wire": [["0001.request.json", 10]]}, separators=(",", ":")).encode()
    (folder / "blobs.json").write_bytes(manifest)
    metadata = {
        "schema": resume.CHECKPOINT_SCHEMA,
        "seq": 1,
        "sends": 0,
        "state_bytes": len(state),
        "state_sha256": hashlib.sha256(state).hexdigest(),
        "blob_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    (folder / "checkpoint.json").write_text(json.dumps(metadata))

    selected = resume.select_checkpoint(run, [{"send": 1}], tmp_path / "wire")

    assert selected is None
    events = [json.loads(line) for line in (checkpoints / "events.jsonl").read_text().splitlines()]
    refusal = next(event for event in events if event["event"] == "checkpoint_restore_refused")
    assert "missing pooled blob wire/0001.request.json" in refusal["detail"]
