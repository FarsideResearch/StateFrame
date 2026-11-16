"""Stage 6 schema ids: neutral ids are current, legacy-prefixed stored artifacts stay readable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kernel_final_product_manifest as manifest
from tools.sciledger import run_real_binding
from tools.sciledger.final_product import FinalProductError

_LEGACY = "a" "ha-"


def test_current_ids_are_neutral_and_shared() -> None:
    assert manifest.LIVE_SCHEMA == "bug-replay-stage6-live/v1"
    assert manifest.CANDIDATE_SCHEMA == "bug-replay-stage6-candidate-diff/v3"
    assert manifest.LIVE_SCHEMA is run_real_binding.LIVE_SCHEMA
    assert manifest.CANDIDATE_SCHEMA is run_real_binding.CANDIDATE_SCHEMA


@pytest.mark.parametrize("schema", [manifest.CANDIDATE_SCHEMA, _LEGACY + manifest.CANDIDATE_SCHEMA])
def test_schema_matches_accepts_current_and_legacy(schema: str) -> None:
    assert run_real_binding.schema_matches(schema, manifest.CANDIDATE_SCHEMA)


@pytest.mark.parametrize("schema", [
    None, 3, "", "bug-replay-stage6-candidate-diff/v2", "other-" + manifest.CANDIDATE_SCHEMA,
    manifest.LIVE_SCHEMA, _LEGACY + _LEGACY + manifest.CANDIDATE_SCHEMA,
])
def test_schema_matches_rejects_everything_else(schema: object) -> None:
    assert not run_real_binding.schema_matches(schema, manifest.CANDIDATE_SCHEMA)


@pytest.mark.parametrize("prefix", ["", _LEGACY])
def test_candidate_document_reads_current_and_legacy(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "2.json"
    path.write_text(json.dumps({"schema": prefix + manifest.CANDIDATE_SCHEMA, "proposals": 2}),
                    encoding="utf-8")
    assert manifest._candidate_document(path) == {
        "schema": prefix + manifest.CANDIDATE_SCHEMA, "proposals": 2}


def test_candidate_document_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "2.json"
    path.write_text(json.dumps({"schema": "bug-replay-stage6-candidate-diff/v2", "proposals": 2}),
                    encoding="utf-8")
    assert manifest._candidate_document(path) is None


def _live(schema: str) -> dict[str, object]:
    return {
        "schema": schema,
        "tasks": [{"task_id": "t1", "outer_run_id": "run-1", "task_bundle_id": "bundle-1"}],
        "artifact_certificates": {"image_manifest_certificate_sha256": "img-1",
                                  "release_id": "rel-1"},
    }


@pytest.mark.parametrize("prefix", ["", _LEGACY])
def test_stage6_identity_reads_current_and_legacy(tmp_path: Path, prefix: str) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps(_live(prefix + manifest.LIVE_SCHEMA)), encoding="utf-8")
    identity = manifest.stage6_identity(path, task_id="t1", solver_run_id="solver-1")
    assert identity["run_id"] == "run-1"
    assert identity["release_digest"] == "rel-1"


def test_stage6_identity_rejects_candidate_schema(tmp_path: Path) -> None:
    path = tmp_path / "live.json"
    path.write_text(json.dumps(_live(manifest.CANDIDATE_SCHEMA)), encoding="utf-8")
    with pytest.raises(FinalProductError, match="unknown schema"):
        manifest.stage6_identity(path, task_id="t1", solver_run_id="solver-1")
