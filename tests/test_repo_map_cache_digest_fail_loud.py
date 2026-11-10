"""An incomplete repo-map snapshot must never produce a usable cache key."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENGINE = SRC / "stateframe" / "engine"
REASON = "REPO_MAP_CACHE_SNAPSHOT_PATH_OUTSIDE_ROOT"


@pytest.fixture(scope="module")
def repo_map_modules():
    for path in (str(SRC), str(ENGINE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from stateframe import ts_grammars

    ts_grammars.install_as_language_pack()
    return (
        importlib.import_module("kernel_repo_intel"),
        importlib.import_module("mc_repo_intel_vendor.repo_map"),
    )


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "inside.py").write_text(
        "def inside_workspace_marker():\n    return True\n", encoding="utf-8",
    )
    outside = tmp_path / "outside.py"
    outside.write_text("def outside_workspace_marker():\n    return True\n", encoding="utf-8")
    return root, outside


def _inject_out_of_root_entry(monkeypatch: pytest.MonkeyPatch, root: Path, outside: Path) -> None:
    original_rglob = Path.rglob
    injected = False

    def rglob(path: Path, pattern: str):
        nonlocal injected
        if path == root and not injected:
            injected = True
            yield outside
        yield from original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", rglob)


def test_cache_snapshot_digest_returns_reason_coded_unavailable_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repo_map_modules,
) -> None:
    from kernel_obligation_assess import UnavailableEvidence

    _kernel_repo_intel, repo_map_module = repo_map_modules
    root, outside = _workspace(tmp_path)
    _inject_out_of_root_entry(monkeypatch, root, outside)

    result = repo_map_module._cache_snapshot_digest(str(root))

    assert isinstance(result, UnavailableEvidence)
    assert result.reason_code == REASON


def test_unavailable_snapshot_bypasses_cache_and_records_reason_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_map_modules,
) -> None:
    kernel_repo_intel, repo_map_module = repo_map_modules
    root, outside = _workspace(tmp_path)
    _inject_out_of_root_entry(monkeypatch, root, outside)
    cache_home = tmp_path / "stateframe-home"
    cache_home.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(cache_home))
    repo_map, stats = repo_map_module.build_repo_map_with_stats(
        str(root), commit_sha="f" * 40, use_cache=True,
    )

    assert "inside_workspace_marker" in repo_map
    assert "outside_workspace_marker" not in repo_map
    assert stats["source"] == "generated"
    assert not list((cache_home / "repo_map_cache").glob("*.txt"))
    assert stats["cache_identity_reason_code"] == REASON

    from kernel_pilot import ACTIVE_RUNTIME

    events: list[tuple[str, dict[str, object]]] = []
    runtime = SimpleNamespace(journal=SimpleNamespace(
        emit=lambda kind, **fields: events.append((kind, fields)),
    ))
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        kernel_repo_intel._journal_repo_map_receipt(stats, map_chars=len(repo_map))
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert len(events) == 1
    assert events[0][0] == "repo_map_modes"
    assert events[0][1]["cache_identity_reason_code"] == REASON
