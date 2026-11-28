from unittest.mock import Mock

import kernel_repo_catalog as catalog


def test_compare_elides_only_byte_identical_replacements(tmp_path, monkeypatch):
    path = tmp_path / "catalog.json"
    files = {"src/β.py": {"value": 3}}
    replace = Mock(wraps=catalog.os.replace)
    monkeypatch.setattr(catalog.os, "replace", replace)
    monkeypatch.setenv("STATEFRAME_CATALOG_COMPARE_BYTES", "off")
    catalog._save_cache(path, files)
    expected = path.read_bytes()
    catalog._save_cache(path, files)
    assert replace.call_count == 2
    monkeypatch.setenv("STATEFRAME_CATALOG_COMPARE_BYTES", "on")
    catalog._save_cache(path, files)
    assert path.read_bytes() == expected
    assert replace.call_count == 2
    for changed in (expected + b"\n", expected[:-1], b"bad UTF-8 \xff"):
        path.write_bytes(changed)
        catalog._save_cache(path, files)
        assert path.read_bytes() == expected
    assert replace.call_count == 5
