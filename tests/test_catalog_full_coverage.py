"""Structural index coverage vs. render budget (STATEFRAME_CATALOG_FULL_COVERAGE).

The index must cover every eligible file; what reaches the model stays within the
legacy MAX_FILES budget and is ranked by relevance. With the switch off, or for a
repository that fits the budget, the rendered bytes are the legacy bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import kernel_repo_catalog as catalog
from kernel_repo_catalog import (
    FULL_COVERAGE_ENV,
    MAX_FILES,
    MAX_INDEX_FILES,
    StructuralIndexError,
    build_structural_index,
    full_coverage_enabled,
    rank_render_paths,
    render_budget,
    render_structural_index,
)

# Byte digest of the index view for ``_PARITY_FILES`` under the pre-switch code path.
_PARITY_DIGEST = "04edf00423f4f953cc24a661b4e2d17a57a88e0fa82c76b8666b25cf34d23eb7"
_PARITY_FILES = {
    "pkg/__init__.py": "",
    "pkg/core.py": "import pkg.util\n\nVALUE = 3\n\n\ndef compute(x):\n    return pkg.util.double(x) + VALUE\n",
    "pkg/util.py": "def double(x):\n    return x * 2\n",
    "app/main.py": "from pkg.core import compute\n\n\ndef run():\n    return compute(4)\n",
    "app/config.toml": 'name = "demo"\nlimit = 5\n',
    "README.md": "not a source file\n",
}
_BIG_PACKAGES = 10
_BIG_SUBPACKAGES = 50
_BIG_FILES_PER_DIR = 100
_BIG_TOTAL = _BIG_PACKAGES * _BIG_SUBPACKAGES * _BIG_FILES_PER_DIR
_LAST_PATH = f"pkg_{_BIG_PACKAGES - 1:02d}/sub_{_BIG_SUBPACKAGES - 1:02d}/m_{_BIG_FILES_PER_DIR - 1:05d}.py"
_NEIGHBOUR_PATH = "pkg_00/sub_00/m_00001.py"


def _use_cache(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    monkeypatch.setattr(catalog, "_cache_path", lambda _root: cache)


def _view_digest(index: catalog.StructuralIndex) -> str:
    view = index.as_dict()
    view.pop("cache_path", None)
    encoded = json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_tree(root: Path, *, packages: int, subpackages: int, files: int) -> None:
    for package in range(packages):
        for sub in range(subpackages):
            directory = root / f"pkg_{package:02d}" / f"sub_{sub:02d}"
            directory.mkdir(parents=True)
            for index in range(files):
                (directory / f"m_{index:05d}.py").write_text(f"def f_{index}():\n    return {index}\n", encoding="utf-8")


@pytest.fixture(scope="module")
def big_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """50,000 tiny Python files across nested directories, built once per worker."""
    root = tmp_path_factory.mktemp("catalog_big") / "big_repo"
    _write_tree(root, packages=_BIG_PACKAGES, subpackages=_BIG_SUBPACKAGES, files=_BIG_FILES_PER_DIR)
    return root


@pytest.fixture
def cheap_outline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coverage is under test here, not the outliner; a stub keeps 50k files affordable.

    Only ``import X`` lines are reported, which is enough for the import-neighbour tier.
    """

    def _stub(source: str, relpath: str) -> dict[str, object]:
        imports = [{"module": line[len("import "):].strip(), "names": []}
                   for line in source.splitlines() if line.startswith("import ")]
        return {"schema": "mc-outline/1", "file": relpath, "language": "python",
                "parser": "stub", "degraded": False, "imports": imports}

    monkeypatch.setattr(catalog, "outline_source", _stub)


def test_switch_defaults_off_and_rejects_unknown_values() -> None:
    assert full_coverage_enabled({}) is False
    assert full_coverage_enabled({FULL_COVERAGE_ENV: "on"}) is True
    assert full_coverage_enabled({FULL_COVERAGE_ENV: "off"}) is False
    with pytest.raises(StructuralIndexError):
        full_coverage_enabled({FULL_COVERAGE_ENV: "maybe"})


@pytest.mark.parametrize("switch", [None, "on"])
def test_small_repo_rendered_bytes_match_legacy_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, switch: str | None,
) -> None:
    root = tmp_path / "parity_repo"
    for rel, text in _PARITY_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    _use_cache(monkeypatch, tmp_path / "cache.json")
    if switch is not None:
        monkeypatch.setenv(FULL_COVERAGE_ENV, switch)
    index = build_structural_index(root, include_call_graph=False)
    assert index.stats.scan_scope == "full"
    assert _view_digest(index) == _PARITY_DIGEST


def test_over_budget_repo_renders_named_file_and_neighbours_within_budget(
    tmp_path: Path, cheap_outline: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end wiring on a tree just past the budget, so it stays fast."""
    root = tmp_path / "wide_repo"
    _write_tree(root, packages=1, subpackages=1, files=MAX_FILES + 88)
    last = "pkg_00/sub_00/m_00599.py"
    (root / last).write_text("import pkg_00.sub_00.m_00001\n", encoding="utf-8")
    _use_cache(monkeypatch, tmp_path / "cache.json")

    monkeypatch.delenv(FULL_COVERAGE_ENV, raising=False)
    legacy = build_structural_index(root, include_call_graph=False)
    assert legacy.module(last) is None
    assert legacy.stats.scan_scope == "truncated"

    monkeypatch.setenv(FULL_COVERAGE_ENV, "on")
    rendered = build_structural_index(root, include_call_graph=False, priority_paths=[last])
    assert len(rendered.modules) == MAX_FILES
    assert rendered.module(last) is not None
    assert rendered.module(_NEIGHBOUR_PATH) is not None
    assert rendered.stats.scan_scope == "relevance_ranked"
    assert rendered.stats.candidate_files == MAX_FILES + 88
    assert rendered.stats.omitted_files == 88
    assert rendered.stats.priority_files_omitted == 0


def test_big_repo_index_covers_every_file_and_ranks_named_first(
    big_repo: Path, cheap_outline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One 50k build: coverage is unbounded, the render view is ranked and budgeted."""
    _use_cache(monkeypatch, tmp_path / "cache.json")
    monkeypatch.setenv(FULL_COVERAGE_ENV, "on")

    full = build_structural_index(big_repo, include_call_graph=False, rendered=False)
    assert len(full.modules) == _BIG_TOTAL
    assert full.module(_LAST_PATH) is not None
    assert full.stats.candidate_files == _BIG_TOTAL
    assert full.stats.scan_scope == "full_coverage"

    view = render_structural_index(full, priority_paths=[_LAST_PATH], max_files=MAX_FILES)
    assert len(view.modules) == MAX_FILES
    assert view.module(_LAST_PATH) is not None
    assert view.stats.scan_scope == "relevance_ranked"
    assert view.stats.omitted_files == _BIG_TOTAL - MAX_FILES
    assert view.stats.priority_files_omitted == 0


def test_big_repo_legacy_switch_off_still_truncates_in_path_order(
    big_repo: Path, cheap_outline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_cache(monkeypatch, tmp_path / "cache.json")
    monkeypatch.delenv(FULL_COVERAGE_ENV, raising=False)
    index = build_structural_index(big_repo, include_call_graph=False)
    assert len(index.modules) == MAX_FILES
    assert index.stats.scan_scope == "truncated"
    assert index.module(_LAST_PATH) is None


def test_rank_orders_named_then_import_neighbours_then_path_order() -> None:
    full_index = _tiny_index(
        {
            "pkg_09/sub_49/m_00099.py": ("pkg_00.sub_00.m_00001",),
            _NEIGHBOUR_PATH: (),
            "pkg_00/sub_00/m_00002.py": (),
        }
    )
    ranked = rank_render_paths(full_index, ["pkg_09/sub_49/m_00099.py"], budget=10)
    assert ranked == ("pkg_09/sub_49/m_00099.py", _NEIGHBOUR_PATH, "pkg_00/sub_00/m_00002.py")
    assert rank_render_paths(full_index, ["pkg_09/sub_49/m_00099.py"], budget=1) == (
        "pkg_09/sub_49/m_00099.py",
    )


def test_render_budget_never_exceeds_legacy_base_budget() -> None:
    assert render_budget(10_000) == MAX_FILES
    assert render_budget(0) == 1
    assert render_budget(MAX_FILES) <= MAX_INDEX_FILES


def _tiny_index(imports_by_path: dict[str, tuple[str, ...]]) -> catalog.StructuralIndex:
    """Minimal index whose modules carry only the path and import fields ranking reads."""
    from kernel_catalog_contracts import ContentHandle, CoverageRef, CoverageStatus, Namespace

    namespace = Namespace("workspace:test", "tiny")
    root = Path("/nonexistent-root-for-ranking")
    modules = []
    for path, imports in sorted(imports_by_path.items()):
        body = b"x = 1\n"
        content = ContentHandle.capture(body, namespace=namespace, source_ref=path)
        coverage = CoverageRef(f"coverage:{path}", "structural_index", CoverageStatus.COMPLETE,
                               inspected=len(body), expected=len(body))
        modules.append(catalog.ModuleRecord(
            f"module:{path}", path, "python", "stub", False, content, coverage, imports,
        ))
    return catalog.StructuralIndex(
        root=root, namespace=namespace, modules=tuple(modules), configs=(), types=(), spans=(),
        dependencies=(), coverage={}, contents={}, bodies={}, outlines={},
        cache_path=root / "cache.json",
        stats=catalog.IndexStats(0, 0, 0, (), 0, candidate_files=len(modules), indexed_files=len(modules)),
    )
