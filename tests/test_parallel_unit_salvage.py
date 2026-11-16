"""An unsubmitted subframe's edits survive in the parent call root.

Live P0 (--parallel 2, 0.7.9): a subframe ended HARNESS_CONTRACT after its replan was
refused; the parent recorded REFUSED with no files, and its edits were left only in
the unit's private worker directory.
"""
from __future__ import annotations

import hashlib
import json
import os

from kernel_parallel import preserve_unit_candidate


def _bundle(root, unit, index, patch, files=("pricing.py",)):
    path = root / "units" / unit / "worker" / "candidates"
    path.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"patch": patch, "file_hashes": {name: "0" * 64 for name in files},
                      "files_base64": {}, "capture": {}, "deleted_files": [],
                      "proposals": [], "target": files[0]}).encode()
    (path / f"{index:04d}.json").write_bytes(raw)
    return raw


def test_newest_edit_bundle_is_preserved_byte_for_byte(tmp_path) -> None:
    _bundle(tmp_path, "pricing", 10, "--- a\n+++ b\n@@ old\n")
    newest = _bundle(tmp_path, "pricing", 11, "--- a/pricing.py\n+++ b/pricing.py\n@@ new\n")

    receipt = preserve_unit_candidate(tmp_path, "pricing")

    assert receipt == {"status": "PRESERVED", "path": os.path.join("salvage", "pricing.json"),
                       "source": "0011.json", "sha256": hashlib.sha256(newest).hexdigest(),
                       "files": ["pricing.py"]}
    assert (tmp_path / "salvage" / "pricing.json").read_bytes() == newest
    assert not [p for p in (tmp_path / "salvage").iterdir() if p.name.endswith(".tmp")]


def test_unit_without_edits_or_bundles_preserves_nothing(tmp_path) -> None:
    assert preserve_unit_candidate(tmp_path, "missing") is None
    _bundle(tmp_path, "clean", 1, "")
    assert preserve_unit_candidate(tmp_path, "clean") is None
    assert not (tmp_path / "salvage").exists()


def test_symlinked_or_corrupt_bundle_never_raises(tmp_path) -> None:
    folder = tmp_path / "units" / "evil" / "worker" / "candidates"
    folder.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"patch": "x"}))
    (folder / "0001.json").symlink_to(outside)
    assert preserve_unit_candidate(tmp_path, "evil")["status"] == "REFUSED"

    _bundle(tmp_path, "bad", 1, "x")
    (tmp_path / "units" / "bad" / "worker" / "candidates" / "0002.json").write_text("{not json")
    assert preserve_unit_candidate(tmp_path, "bad")["status"] == "FAILED"
    assert not (tmp_path / "salvage" / "evil.json").exists()
