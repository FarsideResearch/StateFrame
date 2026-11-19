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
                       "files": ["pricing.py"], "files_total": 1}
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

    folder = tmp_path / "units" / "bad" / "worker" / "candidates"
    folder.mkdir(parents=True)
    (folder / "0002.json").write_text("{not json")
    assert preserve_unit_candidate(tmp_path, "bad") == {"status": "FAILED",
                                                        "reason": "no readable candidate bundle"}
    assert not (tmp_path / "salvage" / "evil.json").exists()


def _compact(root, unit, index, files, *, target, deleted=(), patch=""):
    path = root / "units" / unit / "worker" / "candidates"
    path.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"patch": patch, "target": target, "capture": "stateframe-compact/1",
                      "file_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
                      "files_base64": {}, "deleted_files": list(deleted), "proposals": index}).encode()
    (path / f"{index:04d}.json").write_bytes(raw)
    return raw


def _start(root, files, generation=0):
    base = root / "snapshots" / f"g{generation}"
    for name, data in files.items():
        (base / name).parent.mkdir(parents=True, exist_ok=True)
        (base / name).write_bytes(data)


def test_non_target_edit_and_deletion_are_salvaged(tmp_path) -> None:
    # The bundle's patch is the diff of its single target file. An edit to another
    # owned file, or a deletion, leaves it empty.
    start = {"pricing.py": b"RATE = 1\n", "tests/test_pricing.py": b"assert True\n"}
    _start(tmp_path, start)
    edited = _compact(tmp_path, "other", 4, {**start, "tests/test_pricing.py": b"assert RATE\n"},
                      target="pricing.py")
    receipt = preserve_unit_candidate(tmp_path, "other", start_generation=0)
    assert (receipt["status"], receipt["files"]) == ("PRESERVED", ["tests/test_pricing.py"])
    assert (tmp_path / "salvage" / "other.json").read_bytes() == edited

    _compact(tmp_path, "deleter", 2, {"pricing.py": start["pricing.py"]}, target="pricing.py",
             deleted=["tests/test_pricing.py"])
    assert preserve_unit_candidate(tmp_path, "deleter", start_generation=0)["files"] == ["tests/test_pricing.py"]

    _compact(tmp_path, "untouched", 2, start, target="pricing.py")
    assert preserve_unit_candidate(tmp_path, "untouched", start_generation=0) is None

    # Without the start snapshot the subframe's first capture stands in for it.
    _compact(tmp_path, "nosnap", 1, start, target="pricing.py")
    _compact(tmp_path, "nosnap", 2, {**start, "tests/test_pricing.py": b"assert RATE\n"}, target="pricing.py")
    assert preserve_unit_candidate(tmp_path, "nosnap")["files"] == ["tests/test_pricing.py"]


def test_truncated_newest_falls_back_to_previous_bundle(tmp_path) -> None:
    whole = _bundle(tmp_path, "killed", 7, "--- a/pricing.py\n+++ b/pricing.py\n@@ kept\n")
    (tmp_path / "units" / "killed" / "worker" / "candidates" / "0008.json").write_bytes(whole[:40])

    receipt = preserve_unit_candidate(tmp_path, "killed")

    assert (receipt["status"], receipt["source"]) == ("PRESERVED", "0007.json")
    assert (tmp_path / "salvage" / "killed.json").read_bytes() == whole


def test_salvage_failure_never_aborts_the_call(tmp_path, monkeypatch) -> None:
    folder = tmp_path / "units" / "deep" / "worker" / "candidates"
    folder.mkdir(parents=True)
    (folder / "0001.json").write_text("[" * 200000)  # the decoder raises RecursionError
    assert preserve_unit_candidate(tmp_path, "deep") == {"status": "FAILED",
                                                         "reason": "no readable candidate bundle"}

    _bundle(tmp_path, "boom", 1, "x")

    def explode(*_args, **_kwargs):
        raise RuntimeError(f"cannot write under {tmp_path}")

    monkeypatch.setattr("kernel_parallel.tempfile.mkstemp", explode)
    assert preserve_unit_candidate(tmp_path, "boom") == {"status": "FAILED", "reason": "RuntimeError"}


def test_five_digit_bundle_orders_numerically(tmp_path) -> None:
    _bundle(tmp_path, "long", 9999, "--- a\n+++ b\n@@ older\n")
    newest = _bundle(tmp_path, "long", 10000, "--- a\n+++ b\n@@ newest\n")

    receipt = preserve_unit_candidate(tmp_path, "long")

    assert receipt["source"] == "10000.json"
    assert (tmp_path / "salvage" / "long.json").read_bytes() == newest


def test_journal_file_list_is_bounded_and_names_changed_files_only(tmp_path) -> None:
    start = {f"m{index:03d}.py": b"x = 1\n" for index in range(100)}
    _start(tmp_path, start)
    files = {**{name: b"x = 2\n" for name in start}, "kept.py": b"same\n", "n" * 500 + ".py": b"new\n"}
    _start(tmp_path, {"kept.py": b"same\n"})
    _compact(tmp_path, "wide", 1, files, target="m000.py")

    receipt = preserve_unit_candidate(tmp_path, "wide", start_generation=0)

    assert receipt["files_total"] == 101 and len(receipt["files"]) == 64
    assert "kept.py" not in receipt["files"] and all(len(name) <= 200 for name in receipt["files"])


def test_bundle_swapped_for_symlink_after_the_check_is_refused(tmp_path, monkeypatch) -> None:
    """The open itself refuses a symlink, so a swap after the lstat check cannot redirect the read."""
    import pathlib

    folder = tmp_path / "units" / "racer" / "worker" / "candidates"
    folder.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"patch": "--- a\n+++ b\n", "target": "pricing.py",
                                   "file_hashes": {}}))
    (folder / "0001.json").symlink_to(outside)
    # Simulate the race: the path looked like a plain file when it was checked.
    monkeypatch.setattr(pathlib.Path, "is_symlink", lambda self: False)

    receipt = preserve_unit_candidate(tmp_path, "racer")

    assert receipt["status"] == "REFUSED"
    assert not (tmp_path / "salvage" / "racer.json").exists()
