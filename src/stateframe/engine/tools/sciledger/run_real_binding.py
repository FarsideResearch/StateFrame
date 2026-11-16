"""real-task runner candidate selection bound to the immutable final-product manifest."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from .final_product import FinalProductError, read_manifest


LIVE_SCHEMA = "bug-replay-stage6-live/v1"
CANDIDATE_SCHEMA = "bug-replay-stage6-candidate-diff/v3"
# Artifacts stored by earlier releases carry a project prefix on the same ids.
# They stay readable; new artifacts use the unprefixed ids above.
_LEGACY_SCHEMA_PREFIX = "a" "ha-"


def schema_matches(value: object, current: str) -> bool:
    """True for the current schema id or its legacy-prefixed form (read-side only)."""
    return isinstance(value, str) and value in (current, _LEGACY_SCHEMA_PREFIX + current)


def select_candidate_boundary(run_dir: Path) -> Path:
    """Select the manifest-named candidate; proposal index and mtime are not authority."""

    manifest_path = run_dir / "final-product-manifest.json"
    try:
        manifest = read_manifest(manifest_path)
    except (FinalProductError, OSError) as exc:
        raise FinalProductError(f"RUNREAL requires a valid final-product manifest: {exc}") from exc
    boundary_name = manifest.candidate_boundary
    if boundary_name is None:
        raise FinalProductError("final-product manifest does not name a candidate boundary")
    candidate = (run_dir / boundary_name).resolve()
    if not candidate.is_relative_to(run_dir.resolve()) or candidate.parent.name != "candidates":
        raise FinalProductError("manifest candidate boundary escapes the run candidates directory")
    if not candidate.is_file() or candidate.is_symlink():
        raise FinalProductError("manifest candidate boundary is missing or not a regular file")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalProductError("manifest candidate boundary is not valid JSON") from exc
    if not isinstance(document, Mapping) or not schema_matches(document.get("schema"), CANDIDATE_SCHEMA):
        raise FinalProductError("manifest candidate boundary has an obsolete schema")
    proposal = document.get("proposals")
    if type(proposal) is not int or proposal < 0 or candidate.stem != str(proposal):
        raise FinalProductError("manifest candidate boundary index does not match its document")
    return candidate


__all__ = ["CANDIDATE_SCHEMA", "LIVE_SCHEMA", "schema_matches", "select_candidate_boundary"]
