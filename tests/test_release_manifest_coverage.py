"""release-manifest.json lists exactly the engine's Python files, each at its current hash.

A new engine module once shipped without a manifest entry while the regen check
stayed green, because the hash refresh only touched files already listed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"


def manifest_drift(engine: Path) -> dict[str, list[str]]:
    listed = json.loads((engine / "release-manifest.json").read_text())["files"]
    on_disk = {
        path.relative_to(engine).as_posix()
        for path in engine.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    return {
        "unlisted": sorted(on_disk - set(listed)),
        "missing": sorted(set(listed) - on_disk),
        "stale": sorted(
            name for name in on_disk & set(listed)
            if listed[name] != hashlib.sha256((engine / name).read_bytes()).hexdigest()
        ),
    }


def test_manifest_lists_every_engine_python_file_at_its_hash():
    drift = manifest_drift(ENGINE)
    print(f"expected={len(json.loads((ENGINE / 'release-manifest.json').read_text())['files'])} "
          f"violated={sum(map(len, drift.values()))}")
    assert drift == {"unlisted": [], "missing": [], "stale": []}


def test_manifest_drift_detects_planted_unlisted_missing_and_stale(tmp_path):
    engine = tmp_path / "engine"
    (engine / "pkg").mkdir(parents=True)
    (engine / "kept.py").write_text("x = 1\n")
    (engine / "pkg" / "edited.py").write_text("y = 1\n")
    files = {name: hashlib.sha256((engine / name).read_bytes()).hexdigest()
             for name in ("kept.py", "pkg/edited.py")}
    files["gone.py"] = "0" * 64
    (engine / "release-manifest.json").write_text(json.dumps({"files": files}))
    (engine / "pkg" / "edited.py").write_text("y = 2\n")
    (engine / "new_module.py").write_text("z = 1\n")
    (engine / "__pycache__").mkdir()
    (engine / "__pycache__" / "ignored.py").write_text("")

    assert manifest_drift(engine) == {
        "unlisted": ["new_module.py"], "missing": ["gone.py"], "stale": ["pkg/edited.py"]}
