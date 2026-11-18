"""Optional parser-pack availability is fixed for one engine interpreter."""
from __future__ import annotations

import builtins
import importlib.util
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import ModuleType, SimpleNamespace

import pytest

from mc_repo_intel_vendor import ts_dispatch


def _load_dispatch(monkeypatch: pytest.MonkeyPatch, probe):
    """Load fresh run state without changing the dispatch module used by other tests."""
    original_import = builtins.__import__

    def controlled_import(name, *args, **kwargs):
        if name == "tree_sitter_language_pack":
            return probe()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    monkeypatch.delitem(sys.modules, "tree_sitter_language_pack", raising=False)
    name = "mc_repo_intel_vendor._probe_dispatch"
    spec = importlib.util.spec_from_file_location(name, ts_dispatch.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("error_type", [ModuleNotFoundError, ImportError])
def test_unavailable_pack_is_probed_once_per_run(monkeypatch, error_type):
    sources = [
        ("pkg/valid.py", "import os\nclass Service:\n    def run(self):\n        return 1\n"),
        ("pkg/empty.py", ""),
        ("pkg/broken.py", "def unfinished(:\n"),
        ("script", "#!/usr/bin/python3\r\ndef café():\r\n    return 'é'\r\n"),
    ]
    # Each fresh module represents the fresh interpreter that owns an engine run.
    for _ in range(2):
        attempts = []

        def unavailable():
            attempts.append("probe")
            raise error_type("optional parser pack unavailable")

        with monkeypatch.context() as patch:
            dispatch = _load_dispatch(patch, unavailable)
            expected = [dispatch._python_ast_fallback(
                source, path, dispatch._detected_language(path, source)[1],
            ) for path, source in sources]
            reports = []
            token = dispatch.bind_outline_capability_listener(lambda *row: reports.append(row))
            try:
                actual = [dispatch.outline_source(source, path)
                          for path, source in sources for _ in range(2)]
            finally:
                dispatch.reset_outline_capability_listener(token)

            assert len(attempts) == 1, "pack import must be probed once per engine run"
            assert actual == [outline for outline in expected for _ in range(2)]
            assert len(reports) == len(actual)
            assert actual[0]["types"][0]["name"] == "Service"
            assert actual[0]["types"][0]["methods"][0]["name"] == "run"
            assert actual[-1]["functions"][0]["name"] == "café"


def test_missing_pack_python_outlines_bypass_parser_after_probe(monkeypatch):
    def unavailable():
        raise ModuleNotFoundError("optional parser pack unavailable")

    dispatch = _load_dispatch(monkeypatch, unavailable)
    parse_source = dispatch._parse_source
    attempts = []

    def counted_parse(language, source):
        attempts.append(language)
        return parse_source(language, source)

    monkeypatch.setattr(dispatch, "_parse_source", counted_parse)
    for index in range(4):
        outline = dispatch.outline_source("def run(): pass", f"module{index}.py")
        assert outline["parser"] == "ast"
    assert attempts == ["python"], "known missing pack must go directly to the AST fallback"


def test_concurrent_first_outlines_share_one_missing_pack_probe(monkeypatch):
    attempts = []
    start = Barrier(8)

    def unavailable():
        attempts.append("probe")
        time.sleep(0.05)
        raise ModuleNotFoundError("optional parser pack unavailable")

    dispatch = _load_dispatch(monkeypatch, unavailable)

    def outline(index):
        start.wait(timeout=10)
        return dispatch.outline_source("def run(): pass", f"module{index}.py")

    with ThreadPoolExecutor(max_workers=8) as pool:
        outlines = list(pool.map(outline, range(8)))
    assert [row["parser"] for row in outlines] == ["ast"] * 8
    assert attempts == ["probe"]


@pytest.mark.parametrize("error_type", [ModuleNotFoundError, ImportError])
def test_missing_pack_is_shared_by_outline_and_capability_paths(monkeypatch, error_type):
    attempts = []

    def unavailable():
        attempts.append("probe")
        raise error_type("optional parser pack unavailable")

    dispatch = _load_dispatch(monkeypatch, unavailable)
    for _ in range(3):
        assert ":ast:AVAILABLE:stdlib:" in dispatch.outline_capability_key("app.py")
        assert f":DEGRADED:{error_type.__name__}:" in dispatch.outline_capability_key("app.ts")
        outline = dispatch.outline_source("function load() {}", "app.ts")
        assert outline == dispatch._lexical_outline(
            "function load() {}", "app.ts", "typescript", "extension", error_type.__name__,
        )
        assert dispatch.structural_spans("def run(): pass", "app.py") == {
            "language": "python", "parser": "unavailable", "degraded": True,
            "detection": "extension", "spans": [], "reason": error_type.__name__,
        }
        vue = dispatch.outline_source(
            '<template><Panel /></template>\n<script lang="ts">function load() {}</script>',
            "Panel.vue",
        )
        assert vue["script_blocks"][0]["outline"]["degraded_reason"] == error_type.__name__
        assert "vue-grammar-unavailable" in vue["degraded_reason"]
        assert dispatch.outline_source('{"key": 1}', "data.json") == dispatch._lexical_outline(
            '{"key": 1}', "data.json", "unknown", "fallback", "unsupported-language",
        )
        assert dispatch.dispatch_boundaries_in_file("bus.emit('event')", "app.ts") == []
    assert len(attempts) == 1


def test_available_pack_keeps_parsing_each_file_and_isolates_grammar_failures(monkeypatch):
    attempts = []
    parsed = []
    grammars = []
    pack = ModuleType("tree_sitter_language_pack")
    pack.__version__ = "test"
    root = SimpleNamespace(type="module", named_children=[], children=[], has_error=False)

    def parse(source):
        parsed.append(source)
        return SimpleNamespace(root_node=root)

    def get_parser(language):
        grammars.append(language)
        if language == "go":
            raise LookupError("grammar unavailable")
        return SimpleNamespace(parse=parse)

    pack.get_parser = get_parser

    def available():
        attempts.append("probe")
        return pack

    dispatch = _load_dispatch(monkeypatch, available)
    first = dispatch.outline_source("# first", "first.py")
    missing = dispatch.outline_source("package main", "main.go")
    second = dispatch.outline_source("# second", "second.py")
    assert first["parser"] == second["parser"] == "tree-sitter"
    assert missing["degraded_reason"] == "LookupError"
    assert parsed == [b"# first", b"# second"]
    assert grammars == ["python", "go", "python"]
    assert len(attempts) == 1
