"""The shipped engine manifest keeps its product format discriminator."""
from __future__ import annotations

import json
from pathlib import Path


def test_release_manifest_has_product_engine_format():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "stateframe" / "engine" / "release-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest.get("format") == "stateframe-product-engine/1"
