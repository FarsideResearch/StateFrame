"""The engine switch registry ships precomputed; a run hashes the sources instead of AST-parsing them.

AST-parsing every declared engine file took 22-24 s per scan on a loaded host and ran twice per run
(host and engine child), which pushed first-request waits past their limits.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
sys.path.insert(0, str(ENGINE))
import kernel_release_switch_registry as registry  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    registry._release_registry.cache_clear()
    yield
    registry._release_registry.cache_clear()


def _toy_root(tmp_path: Path) -> Path:
    root = tmp_path / "engine"
    root.mkdir()
    (root / "a.py").write_text('import os\nX = os.environ.get("STATEFRAME_ALPHA")\n')
    (root / "b.py").write_text('import os\nif os.getenv("STATEFRAME_BETA") == "on":\n    pass\n')
    (root / "release-manifest.json").write_text(json.dumps({"files": {"a.py": "", "b.py": ""}}))
    return root


def _forbid_scan(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("source scan ran although the shipped registry matches")
    monkeypatch.setattr(registry, "_scan", refuse)


def test_shipped_registry_matches_a_fresh_scan_of_this_release():
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text())
    expected = registry.build_registry_document(ENGINE)
    assert shipped == expected, (
        f"{registry.REGISTRY_FILE} is stale; regenerate it with "
        "python -I src/stateframe/engine/kernel_release_switch_registry.py --write")


def test_matching_shipped_registry_is_used_without_parsing(tmp_path, monkeypatch):
    root = _toy_root(tmp_path)
    registry.write_registry(root)
    scanned = json.loads((root / registry.REGISTRY_FILE).read_text())
    assert scanned["switches"] == ["STATEFRAME_ALPHA", "STATEFRAME_BETA"]
    registry._release_registry.cache_clear()
    _forbid_scan(monkeypatch)
    switches, digest = registry.release_stateframe_switches(root)
    assert switches == frozenset({"STATEFRAME_ALPHA", "STATEFRAME_BETA"}) and digest == scanned["digest"]


def test_shipped_registry_fingerprint_matches_the_real_engine_sources():
    shipped = json.loads((ENGINE / registry.REGISTRY_FILE).read_text())
    files = registry._declared_python_files(ENGINE)
    assert shipped["sources_sha256"] == registry._sources_fingerprint(ENGINE, files)


def test_the_real_engine_starts_without_parsing_its_sources(monkeypatch):
    _forbid_scan(monkeypatch)
    switches, digest = registry.release_stateframe_switches(ENGINE)
    assert len(switches) > 100 and digest.startswith("sha256:")


def test_host_only_env_seed_setting_is_not_in_engine_switch_registry():
    switches, _ = registry.release_stateframe_switches(ENGINE)
    assert "STATEFRAME_ENV_SEED" not in switches


@pytest.mark.parametrize("damage", ["source_changed", "switches_edited", "digest_edited", "schema", "garbage",
                                    "missing"])
def test_a_stale_or_damaged_registry_falls_back_to_the_scan(tmp_path, damage):
    root = _toy_root(tmp_path)
    registry.write_registry(root)
    path = root / registry.REGISTRY_FILE
    document = json.loads(path.read_text())
    if damage == "source_changed":
        (root / "b.py").write_text('import os\nos.environ.get("STATEFRAME_GAMMA")\n')
    elif damage == "switches_edited":
        document["switches"] = ["STATEFRAME_ALPHA", "STATEFRAME_BETA", "STATEFRAME_FORGED"]
        path.write_text(json.dumps(document))
    elif damage == "digest_edited":
        document["digest"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(document))
    elif damage == "schema":
        document["schema"] = "something-else/1"
        path.write_text(json.dumps(document))
    elif damage == "garbage":
        path.write_text("{not json")
    else:
        path.unlink()
    registry._release_registry.cache_clear()
    switches, digest = registry.release_stateframe_switches(root)
    expected = {"STATEFRAME_ALPHA", "STATEFRAME_GAMMA"} if damage == "source_changed" else {
        "STATEFRAME_ALPHA", "STATEFRAME_BETA"}
    assert switches == frozenset(expected)
    assert "STATEFRAME_FORGED" not in switches
    assert digest == registry.build_registry_document(root)["digest"]
