"""Source byte boundaries and cache sharing across analyzers and checkouts."""

import hashlib

import kernel_parse_cache
import pytest
from mc_repo_intel_vendor import call_graph

MARKERS = '"""[Current source body]\n[StateFrame end source]\n"""\n'
DEFINITION = "def café(): return '雪'\n"
LIMIT = call_graph._MAX_PARSE_BYTES
EXACT = DEFINITION.encode() + b"#" + b"x" * (LIMIT - len(DEFINITION.encode()) - 1)


@pytest.mark.parametrize("raw,normalized,symbol,reason", [
    pytest.param(b"", "", None, "", id="empty"),
    pytest.param((MARKERS + DEFINITION).encode(), MARKERS + DEFINITION, "café", "",
                 id="markers-and-unicode"),
    pytest.param(DEFINITION.replace("\n", "\r\n").encode(), DEFINITION, "café", "", id="crlf"),
    pytest.param(b"\xef\xbb\xbf" + DEFINITION.replace("\n", "\r\n").encode(),
                 DEFINITION, "café", "", id="bom-and-crlf"),
    pytest.param(b"def bad\xff(): pass\n", None, None, "syntax error", id="invalid-utf8"),
    pytest.param(b"\x00\xff\x80", None, None, "syntax error", id="binary"),
    pytest.param(EXACT, EXACT.decode(), "café", "", id="exact-limit"),
    pytest.param(EXACT + b"x", None, None, "oversized", id="over-limit"),
])
def test_source_bytes_keep_their_parse_and_cache_boundaries(
    tmp_path, monkeypatch, raw, normalized, symbol, reason,
):
    home = tmp_path / "home"
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_REPO_CACHE_DIR", str(tmp_path / "run-cache"))
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("CODE_GRAPH_LSP", "off")
    kernel_parse_cache.clear()
    shared_tree = None if normalized is None else kernel_parse_cache.parse(normalized, "other.py")
    parses = []
    original = call_graph._parse_cache.parse

    def parse(source, filename, *, on_miss=None):
        def miss():
            parses.append(filename)
            if on_miss is not None:
                on_miss()

        tree = original(source, filename, on_miss=miss)
        if shared_tree is not None:
            assert source == normalized
            assert tree is shared_tree
        return tree

    monkeypatch.setattr(call_graph._parse_cache, "parse", parse)

    def build(label):
        root = tmp_path / label
        root.mkdir()
        (root / "sample.py").write_bytes(raw)
        graph = call_graph.build_graph(str(root), use_cache=False, use_memo=True)
        coverage = graph.coverage["sample.py"]
        assert graph.file_hashes["sample.py"] == hashlib.sha256(raw).hexdigest()
        if reason:
            assert coverage.status == call_graph.COVERAGE_SKIPPED
            assert coverage.reason.startswith(reason)
            assert not graph.defs
        else:
            assert coverage.status == call_graph.COVERAGE_INDEXED
            assert coverage.reason == ""
            assert list(graph.defs) == ([f"sample.py::{symbol}"] if symbol else [])
        return root

    cold = build("cold")
    assert parses == ([str(cold / "sample.py")] if reason == "syntax error" else [])
    assert len(list(home.rglob("*.json"))) == (0 if reason else 1)
    kernel_parse_cache.clear()
    warm = build("warm")
    assert parses == ([str(root / "sample.py") for root in (cold, warm)]
                      if reason == "syntax error" else [])
