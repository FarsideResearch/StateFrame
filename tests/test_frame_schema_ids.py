"""Frame schema ids are neutral; capture exports from earlier releases stay accepted."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import kernel_adversary
import kernel_fl_checkpoint_capture as capture
import kernel_idealpath
import kernel_s1_spine

_OLD = "frame-l" "ab"


def _s1_compile_schema(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.setattr(kernel_s1_spine, "compile_repro_templates", lambda _ledger, _bindings: {
        "ready_count": 1, "gap_count": 0, "complete": True})
    tasks = [{"task_id": f"t{index}", "ledger": {}, "bindings": {}} for index in range(11)]
    return kernel_s1_spine.frame_lab_compile_check(tasks)["schema"]


def test_current_frame_schema_ids_are_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = (capture._MANIFEST_SCHEMA, capture._CAPTURE_DOCUMENT_SCHEMA,
           kernel_idealpath.FRAMEBUILD_RECEIPT_SCHEMA,
           kernel_adversary.FRAME_LAB_REFERENCE["schema"],
           _s1_compile_schema(monkeypatch))
    assert ids == ("frame/checkpoint-manifest/1", "frame/settled-checkpoint/1",
                   "frame/framebuild-owner-receipt/1", "frame-adversary-s3-s5-s6/v1",
                   "frame-s1-compile/1")
    assert not any(_OLD in value for value in ids)


def _manifest(checkpoint_id: str = "cp-1") -> capture._CompatManifest:
    values: dict[str, object] = {}
    for field in dataclasses.fields(capture._CompatManifest):
        if field.name in {"pending_request", "pending_effect", "event_id", "journal_prefix_hash",
                          "requirement_revision", "tree_digest", "kernel_artifact_hash",
                          "requirement_provenance", "admission_grade"}:
            values[field.name] = None
        elif field.name == "journal_cutoff":
            values[field.name] = 0
        elif field.name in {"model_identity", "sampler_identity", "runtime_identity",
                            "effective_flags", "notes_state", "completion_state", "budgets"}:
            values[field.name] = {}
        elif field.type.startswith("tuple"):
            values[field.name] = ()
        else:
            values[field.name] = f"{field.name}-value"
    values["checkpoint_id"] = checkpoint_id
    return capture._CompatManifest(**values)


def _checkpoint() -> capture._CompatCheckpoint:
    return capture._CompatCheckpoint(manifest=_manifest(), request_payload="{}",
                                     public_state={}, kernel_state={}, admission_state={})


def test_export_writes_current_ids(tmp_path: Path) -> None:
    export = capture._CompatExporter(tmp_path).export(_checkpoint())
    assert export.document["schema"] == "frame/settled-checkpoint/1"
    assert export.document["checkpoint"]["schema"] == "frame/checkpoint-manifest/1"
    assert _OLD.encode() not in export.manifest_path.read_bytes()


def test_export_accepts_identical_legacy_document_on_disk(tmp_path: Path) -> None:
    exporter = capture._CompatExporter(tmp_path)
    current = exporter.export(_checkpoint())
    legacy = capture._legacy_document(current.document)
    assert legacy["schema"] == _OLD + "/settled-checkpoint/1"
    assert legacy["checkpoint"]["schema"] == _OLD + "/checkpoint-manifest/1"
    legacy_raw = capture._canonical(legacy)
    current.manifest_path.write_bytes(legacy_raw)

    again = exporter.export(_checkpoint())
    assert again.manifest_path.read_bytes() == legacy_raw
    assert again.document["schema"] == legacy["schema"]
    assert again.artifact_hash == capture._digest_bytes(legacy_raw)


def test_export_still_refuses_a_different_document(tmp_path: Path) -> None:
    exporter = capture._CompatExporter(tmp_path)
    current = exporter.export(_checkpoint())
    current.manifest_path.write_bytes(b'{"schema":"something-else"}')
    with pytest.raises(capture.CaptureBoundaryError, match="immutable export collision"):
        exporter.export(_checkpoint())
