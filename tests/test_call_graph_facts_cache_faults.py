"""Cache failures preserve graph results and leave an operator-visible reason."""

import json
import logging
from pathlib import Path

import pytest
from mc_repo_intel_vendor import call_graph, lsp_resolve


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sample.py").write_text(
        "def target(): return 1\ndef caller(): return target()\n", encoding="utf-8",
    )
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "run-cache"))
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    return root


def _graph(root):
    call_graph.clear_build_memo()
    graph = call_graph.build_graph(str(root), use_cache=False, use_memo=True)
    assert [ref.symbol for ref in graph.find_callees("caller")] == ["sample.py::target"]
    return graph


def _events(caplog):
    return [(record.cache_event, record.cache_reason) for record in caplog.records
            if hasattr(record, "cache_event")]


@pytest.mark.parametrize("condition,reason", [
    ("inside-checkout", "inside_checkout"),
    ("resolution-error", "PermissionError"),
])
def test_invalid_installation_cache_location_reports_fallback(
    checkout, tmp_path, monkeypatch, caplog, condition, reason,
):
    home = checkout / "home" if condition == "inside-checkout" else tmp_path / "blocked-home"
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    if condition == "resolution-error":
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            if path == home / "repo-intel" / "python-call-facts":
                raise PermissionError("injected cache path failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolve)
    caplog.set_level(logging.DEBUG, logger=call_graph.__name__)
    _graph(checkout)
    assert _events(caplog) == [("python_facts_directory_fallback", reason)]
    assert not (home / "repo-intel").exists()
    assert len(list((tmp_path / "run-cache").rglob("*.json"))) == 1


@pytest.mark.parametrize("damage,reason", [
    ("read-error", "PermissionError"),
    ("invalid-utf8", "UnicodeDecodeError"),
    ("invalid-json", "JSONDecodeError"),
    ("invalid-extract", "invalid_extract"),
    ("missing-imports", "invalid_imports"),
    ("invalid-row", "invalid_imports"),
    ("invalid-level", "invalid_imports"),
    ("invalid-type", "invalid_imports"),
    ("invalid-name", "invalid_imports"),
])
def test_bad_facts_entry_is_reported_and_rebuilt(
    checkout, tmp_path, monkeypatch, caplog, damage, reason,
):
    _graph(checkout)
    entry, = (tmp_path / "home").rglob("*.json")
    good = entry.read_bytes()
    if damage == "read-error":
        original = Path.read_text
        reads = []

        def read(path, *args, **kwargs):
            if path == entry:
                reads.append(path)
                raise PermissionError("injected facts read failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    elif damage == "invalid-utf8":
        entry.write_bytes(b"\xff")
    elif damage == "invalid-json":
        entry.write_bytes(b"{")
    else:
        data = json.loads(good)
        if damage == "invalid-extract":
            data["hash"] = "0" * 64
        elif damage == "missing-imports":
            del data["imports"]
        else:
            rows = {
                "invalid-row": [0],
                "invalid-level": [-1, "pkg", "symbol", ""],
                "invalid-type": [0, "pkg", None, ""],
                "invalid-name": [0, "pkg", "../symbol", ""],
            }
            data["imports"] = [rows[damage]]
        entry.write_text(json.dumps(data), encoding="utf-8")
    caplog.set_level(logging.DEBUG, logger=call_graph.__name__)
    _graph(checkout)
    assert _events(caplog) == [("python_facts_read_failure", reason)]
    assert entry.read_bytes() == good
    if damage == "read-error":
        assert reads == [entry]


def test_missing_facts_cold_build_is_observable(checkout, tmp_path, monkeypatch, caplog):
    original = call_graph._extract_python_facts
    extracted = []

    def extract(source, relpath, filename):
        extracted.append(relpath)
        return original(source, relpath, filename)

    monkeypatch.setattr(call_graph, "_extract_python_facts", extract)
    caplog.set_level(logging.DEBUG, logger=call_graph.__name__)
    _graph(checkout)
    entry, = (tmp_path / "home").rglob("*.json")
    assert json.loads(entry.read_bytes())["extract"]["defs"]
    assert extracted == ["sample.py"]
    assert not _events(caplog)
    _graph(checkout)
    assert extracted == ["sample.py"]


def test_failed_facts_publish_cleans_up_and_keeps_the_graph(
    checkout, tmp_path, monkeypatch, caplog,
):
    original = call_graph.os.replace
    files = []

    def replace(source, target):
        if "python-call-facts" in Path(target).parts:
            assert json.loads(Path(source).read_bytes())["extract"]["defs"]
            files.append((Path(source), Path(target)))
            raise PermissionError("injected facts publish failure")
        return original(source, target)

    caplog.set_level(logging.DEBUG, logger=call_graph.__name__)
    with monkeypatch.context() as patch:
        patch.setattr(call_graph.os, "replace", replace)
        _graph(checkout)
    assert len(files) == 1
    assert all(not path.exists() for path in files[0])
    assert _events(caplog) == [("python_facts_write_failure", "PermissionError")]
    _graph(checkout)
    assert list((tmp_path / "home").rglob("*.json")) == [files[0][1]]


@pytest.mark.parametrize("failure,consulted,message", [
    ("before", False, "workspace changed before deferred type resolution"),
    ("during", True, "workspace changed during deferred type resolution"),
    ("resolver-error", True, "lsp resolution pass failed"),
])
def test_deferred_type_resolution_reports_fallback_and_keeps_static_edges(
    checkout, monkeypatch, caplog, failure, consulted, message,
):
    monkeypatch.setenv("CODE_GRAPH_LSP", "on")
    monkeypatch.setattr(lsp_resolve, "_JEDI_AVAILABLE", True)
    (checkout / "other.py").write_text("def other(): return 2\n")

    def change_tree():
        (checkout / "changed.py").write_text("changed = True\n")

    attempts = []

    def resolve(root, sites_by_file, **kwargs):
        assert root == checkout
        attempts.append(root)
        site, = sites_by_file["sample.py"]
        assert site.name == "target"
        if failure == "resolver-error":
            raise RuntimeError("injected resolver failure")
        change_tree()
        other = lsp_resolve.ResolvedTarget("other.py", 1, "other", "function")
        return {"sample.py": {site.key: other}}, {}

    monkeypatch.setattr(lsp_resolve, "resolve_repo", resolve)
    graph = call_graph.build_graph(str(checkout), use_cache=False, use_memo=True)
    assert graph.find_definition("caller").file == "sample.py"
    assert not attempts
    if failure == "before":
        change_tree()
    caplog.set_level(logging.DEBUG, logger=call_graph.__name__)
    assert [ref.symbol for ref in graph.find_callees("caller")] == ["sample.py::target"]
    assert any(message in record.getMessage() for record in caplog.records)
    assert attempts == ([checkout] if consulted else [])
    assert graph.lsp_stats["overridden"] == 0
