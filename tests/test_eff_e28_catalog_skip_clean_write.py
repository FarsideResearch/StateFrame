import json
from unittest.mock import Mock

import pytest

import kernel_repo_catalog as catalog


@pytest.fixture
def catalog_fixture(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(catalog, "_cache_path", lambda root: cache)
    save = Mock(wraps=catalog._save_cache)
    monkeypatch.setattr(catalog, "_save_cache", save)
    return root, cache, save


def test_clean_focused_outline_does_not_replace_full_cache(monkeypatch, catalog_fixture):
    root, cache, save = catalog_fixture
    for name in ("one.py", "two.py"):
        (root / name).write_text("VALUE = 1\n")
    catalog.build_structural_index(root, include_call_graph=False)
    original = cache.read_bytes()
    receipts = []
    for mode, writes in (("off", 1), ("on", 0)):
        monkeypatch.setenv("STATEFRAME_CATALOG_SKIP_CLEAN_WRITE", mode)
        save.reset_mock()
        view = catalog.build_structural_index(root, focus_paths=["one.py"]).file_view("one.py")
        receipts.append(json.dumps(view, separators=(",", ":")).encode())
        assert save.call_count == writes
        assert cache.read_bytes() == original
    assert receipts[0] == receipts[1]


@pytest.mark.parametrize("initial", [None, b"invalid", b"{}", b'{"files":[]}'])
def test_missing_corrupt_empty_and_changed_catalog_publish_correctly(
        monkeypatch, catalog_fixture, initial):
    monkeypatch.setenv("STATEFRAME_CATALOG_SKIP_CLEAN_WRITE", "on")
    root, cache, save = catalog_fixture
    if initial is not None:
        cache.write_bytes(initial)
    catalog.build_structural_index(root, include_call_graph=False)
    assert json.loads(cache.read_bytes())["files"] == {}
    catalog.build_structural_index(root, include_call_graph=False)
    assert save.call_count == 1
    for value in ("1", "2"):
        (root / "one.py").write_text(f"VALUE = {value}\n")
        index = catalog.build_structural_index(root, focus_paths=["one.py"])
        assert index.stats.rebuilt_files == ("one.py",)
    assert save.call_count == 3
