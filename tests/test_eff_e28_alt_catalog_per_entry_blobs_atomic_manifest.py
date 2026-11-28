import json

import kernel_repo_catalog as catalog


def test_dirty_focus_writes_only_its_entry_and_manifest(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(catalog, "_cache_path", lambda _: cache)
    for i in range(16):
        (root / f"item{i}.py").write_text(f"VALUE = {i}\n# " + "x" * 4096)
    writes = []
    publish = catalog._replace_cache_bytes

    def count(path, data):
        writes.append((path, len(data)))
        publish(path, data)

    monkeypatch.setattr(catalog, "_replace_cache_bytes", count)
    receipts = []
    sizes = []
    for mode in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_CATALOG_ENTRY_BLOBS", mode)
        path = root / "item1.py"
        path.write_text("VALUE = 1\n")
        catalog.build_structural_index(root, include_call_graph=False)
        writes.clear()
        path.write_text("VALUE = 100\n")
        index = catalog.build_structural_index(root, focus_paths=[path.name])
        receipts.append(json.dumps(index.file_view(path.name)).encode())
        sizes.append(sum(size for _, size in writes))
        if mode == "on":
            assert len(writes) == 2
            manifest = writes[-1][0]
            refs = json.loads(manifest.read_bytes())["files"]
            assert len(refs) == 16
            leaf = manifest.with_suffix(".entry-" + refs[path.name])
            leaf.write_bytes(b'{"corrupt":true}')
            repaired = catalog.build_structural_index(root, focus_paths=[path.name])
            assert repaired.stats.rebuilt_files == (path.name,)
            assert catalog.build_structural_index(root, focus_paths=[path.name]).stats.cache_hits == 1
    assert receipts[0] == receipts[1]
    assert sizes[1] < sizes[0] // 10
