"""Fail-loud checks for broad except sites in the repo-intel modules (catalog, intel, call graph, LSP, repo map, tree-sitter).

Scope: kernel_repo_catalog, kernel_repo_intel, mc_repo_intel_vendor/{call_graph, lsp_resolve, repo_map,
ts_dispatch, ts_extract}.

Loudness bar: a raise, a WARNING-or-higher record from the owning module, or a typed field / span / capability
string that a reader checks. Documented fail-open paths and telemetry helpers are degrade-OK and only pinned
where the typed outcome is the safety argument.

T8 tags name the symbol keys of guarded handlers. The three CRASH-HIDER sites on
default paths (catalog callee lookup, TS dispatch scan, has_error probe) were strict xfails; they are now fixed
and asserted, each with a regression test for the typed outcome its caller reads.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_repo_catalog as catalog
import kernel_repo_intel as repo_intel
from _except_sites import (
    UNREVIEWED_HANDLER,
    ExceptSite,
    _broad_except_sites,
    assert_site_tags,
    assert_triaged,
    edited_module_paths,
    shifted_module_paths,
)
from kernel_catalog_contracts import CoverageStatus
from mc_repo_intel_vendor import call_graph, repo_map, ts_dispatch, ts_extract

ENGINE = Path(repo_intel.__file__).resolve().parent
_MODULE_FILES = (
    ENGINE / "kernel_repo_catalog.py",
    ENGINE / "kernel_repo_intel.py",
    ENGINE / "mc_repo_intel_vendor" / "call_graph.py",
    ENGINE / "mc_repo_intel_vendor" / "lsp_resolve.py",
    ENGINE / "mc_repo_intel_vendor" / "repo_map.py",
    ENGINE / "mc_repo_intel_vendor" / "ts_dispatch.py",
    ENGINE / "mc_repo_intel_vendor" / "ts_extract.py",
)


def _loud(caplog: pytest.LogCaptureFixture, module_name: str) -> bool:
    return any(record.levelno >= logging.WARNING and module_name in record.name for record in caplog.records)


class _Ref(SimpleNamespace):
    """Stand-in for a graph callee reference with the ``symbol`` attribute the catalog reads."""


# T8-[kernel_repo_catalog | _graph_dependencies | except Exception as exc: | 2]
# a source whose callee lookup fails must not lose its CALLS edges silently.
def test_graph_dependency_callee_failure_is_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture) -> None:
    class _Graph:
        defs = {"a.py::f": object(), "b.py::g": object()}

        def find_callees(self, source_id: str):
            if source_id == "a.py::f":
                raise RuntimeError("forced: callee lookup")
            return [_Ref(symbol="b.py::h")]

    monkeypatch.setattr(catalog, "build_graph", lambda *a, **k: _Graph())
    coverage = {
        "a.py": catalog.CoverageRef("cov:a", "structural_index", CoverageStatus.COMPLETE),
        "b.py": catalog.CoverageRef("cov:b", "structural_index", CoverageStatus.COMPLETE),
    }
    with caplog.at_level(logging.WARNING, logger=catalog.__name__):
        records = catalog._graph_dependencies(tmp_path, [], coverage)

    assert [r.target_id for r in records] == ["b.py::h"], "healthy source still contributes its edges"
    assert _loud(caplog, "kernel_repo_catalog"), "a dropped source must leave a WARNING+ record"
    assert "a.py::f (RuntimeError)" in caplog.text, "the record names the source and the failure type"


# T8-[call_graph | dispatch_boundaries_in_file | except Exception as exc: | 1]
# a tree-sitter boundary scan failure must read as UNKNOWN boundaries, not "none".
def test_ts_dispatch_scan_failure_is_loud_not_empty(monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    def _boom(source: str, relpath: str):
        raise RuntimeError("forced: ts dispatch scan")

    monkeypatch.setattr(ts_dispatch, "dispatch_boundaries_in_file", _boom)
    with caplog.at_level(logging.WARNING, logger=call_graph.__name__):
        result = call_graph.dispatch_boundaries_in_file("export const a = 1;\n", "web/app.ts")
    assert result != [], "scan failure must not look like an empty scan"
    assert [b.kind for b in result] == [call_graph.SCAN_FAILED_KIND]
    assert result[0].file == "web/app.ts" and "UNKNOWN" in result[0].note and "RuntimeError" in result[0].note
    assert _loud(caplog, "call_graph"), "scan failure must leave a WARNING+ record"


# T8-[ts_extract | _has_error | except Exception as exc: | 1]
# a binding quirk in has_error must not read as "no syntax errors" (file marked INDEXED).
def test_has_error_probe_failure_is_not_read_as_clean(caplog: pytest.LogCaptureFixture) -> None:
    class _Root:
        def has_error(self) -> bool:
            raise RuntimeError("forced: binding quirk")

    with caplog.at_level(logging.WARNING, logger=ts_extract.__name__):
        result = ts_extract._has_error(_Root())
    assert result is True, "unknown error state must not read as clean"
    assert _loud(caplog, "ts_extract"), "probe failure must leave a WARNING+ record"


# ── Regressions for the three fixed sites ─────────────────────────────────────────────────────────

def test_graph_dependencies_report_only_indexed_sources_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    class _Graph:
        defs = {"a.py::f": object(), "a.py::g": object(), "vendor.py::x": object(), "b.py::h": object()}

        def find_callees(self, source_id: str):
            if source_id != "b.py::h":
                raise KeyError(source_id)
            return [_Ref(symbol="a.py::f")]

    monkeypatch.setattr(catalog, "build_graph", lambda *a, **k: _Graph())
    coverage = {
        "a.py": catalog.CoverageRef("cov:a", "structural_index", CoverageStatus.COMPLETE),
        "b.py": catalog.CoverageRef("cov:b", "structural_index", CoverageStatus.COMPLETE),
    }
    with caplog.at_level(logging.WARNING, logger=catalog.__name__):
        records = catalog._graph_dependencies(tmp_path, [], coverage)

    assert [(r.source_id, r.target_id, r.relation) for r in records] == [("b.py::h", "a.py::f", "CALLS")]
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == catalog.__name__]
    assert len(warnings) == 1, "one aggregated record per build, not one per source"
    message = warnings[0].getMessage()
    assert "for 2 source(s)" in message and "a.py::f (KeyError)" in message and "a.py::g (KeyError)" in message
    assert "vendor.py" not in message, "sources outside the index never had edges to lose"


def test_graph_dependencies_all_healthy_stays_quiet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    class _Graph:
        defs = {"a.py::f": object()}

        def find_callees(self, source_id: str):
            return [_Ref(symbol="a.py::g")]

    monkeypatch.setattr(catalog, "build_graph", lambda *a, **k: _Graph())
    coverage = {"a.py": catalog.CoverageRef("cov:a", "structural_index", CoverageStatus.COMPLETE)}
    with caplog.at_level(logging.WARNING, logger=catalog.__name__):
        records = catalog._graph_dependencies(tmp_path, [], coverage)
    assert [r.target_id for r in records] == ["a.py::g"]
    assert not _loud(caplog, "kernel_repo_catalog")


def test_ts_dispatch_module_import_failure_reads_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # A missing ts_dispatch module is the same unknown state as a failed scan.
    monkeypatch.setitem(sys.modules, "mc_repo_intel_vendor.ts_dispatch", None)
    monkeypatch.delattr(sys.modules["mc_repo_intel_vendor"], "ts_dispatch", raising=False)
    result = call_graph.dispatch_boundaries_in_file("export const a = 1;\n", "web/app.js")
    assert [b.kind for b in result] == [call_graph.SCAN_FAILED_KIND]
    assert "ImportError" in result[0].note or "ModuleNotFoundError" in result[0].note


def test_python_dispatch_scan_recursion_keeps_found_boundaries_and_marks_unknown(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    source = "def f(obj, name):\n    return getattr(obj, name)()\n"
    baseline = call_graph.dispatch_boundaries_in_file(source, "pkg/mod.py")
    assert baseline and all(b.kind != call_graph.SCAN_FAILED_KIND for b in baseline)

    original_visit = call_graph._DispatchVisitor.visit
    depth = {"n": 0}

    def _visit_then_overflow(self, node):
        depth["n"] += 1
        try:
            original_visit(self, node)
        finally:
            depth["n"] -= 1
        if depth["n"] == 0:  # overflow only after the whole tree was walked
            raise RecursionError("forced: deep tree")

    monkeypatch.setattr(call_graph._DispatchVisitor, "visit", _visit_then_overflow)
    with caplog.at_level(logging.WARNING, logger=call_graph.__name__):
        result = call_graph.dispatch_boundaries_in_file(source, "pkg/mod.py")
    assert [b.kind for b in result] == [*(b.kind for b in baseline), call_graph.SCAN_FAILED_KIND]
    assert _loud(caplog, "call_graph")


def test_python_parse_overflow_reads_unknown_but_syntax_error_stays_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    assert call_graph.dispatch_boundaries_in_file("def broken(:\n", "pkg/bad.py") == []

    def _overflow(*args, **kwargs):
        raise RecursionError("forced: deep nesting")

    monkeypatch.setattr(call_graph.ast, "parse", _overflow)
    result = call_graph.dispatch_boundaries_in_file("x = 1\n", "pkg/deep.py")
    assert [b.kind for b in result] == [call_graph.SCAN_FAILED_KIND]


def test_scan_failure_marker_is_byte_stable_across_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # The marker reaches agent-facing rows, so it carries only the exception type: no message, path or address.
    messages = iter(("forced: /private/a 0x7f00 first", "forced: second run, other text"))

    def _boom(source: str, relpath: str):
        raise RuntimeError(next(messages))

    monkeypatch.setattr(ts_dispatch, "dispatch_boundaries_in_file", _boom)
    first = call_graph.dispatch_boundaries_in_file("export const a = 1;\n", "web/app.ts")
    second = call_graph.dispatch_boundaries_in_file("export const a = 1;\n", "web/app.ts")
    assert first == second
    assert "private" not in first[0].note and "0x" not in first[0].note


def test_dispatch_boundaries_near_keeps_scan_failure_marker_for_any_range(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "app.ts").write_text("export const a = 1;\n" * 50, encoding="utf-8")

    def _boom(source: str, relpath: str):
        raise RuntimeError("forced: ts dispatch scan")

    monkeypatch.setattr(ts_dispatch, "dispatch_boundaries_in_file", _boom)
    result = call_graph.dispatch_boundaries_near(tmp_path, "web/app.ts", 30, 40)
    assert [b.kind for b in result] == [call_graph.SCAN_FAILED_KIND], "a whole-file unknown covers every range"


def test_has_error_probe_states_stay_exact() -> None:
    assert ts_extract._has_error(SimpleNamespace(has_error=True)) is True
    assert ts_extract._has_error(SimpleNamespace(has_error=False)) is False
    assert ts_extract._has_error(SimpleNamespace(has_error=lambda: False)) is False, "method binding is called"
    assert ts_extract._has_error(SimpleNamespace(has_error=lambda: True)) is True
    assert ts_extract._has_error(SimpleNamespace()) is False


def test_has_error_probe_failure_marks_extract_parse_partial_without_grammars(
        monkeypatch: pytest.MonkeyPatch) -> None:
    class _Root:
        def has_error(self) -> bool:
            raise RuntimeError("forced: binding quirk")

    class _Visitor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.defs: list[object] = []
            self.calls: list[object] = []

        def run(self, root: object) -> str:
            return ""

    monkeypatch.setattr(ts_extract, "is_tree_sitter_functional", lambda: True)
    monkeypatch.setattr(ts_extract, "_tslp", SimpleNamespace(get_parser=lambda lang: object()))
    monkeypatch.setattr(ts_extract, "_parse", lambda parser, data: SimpleNamespace(root_node=_Root()))
    monkeypatch.setattr(ts_extract, "_build_import_table", lambda *args: {})
    monkeypatch.setattr(ts_extract, "_TSVisitor", _Visitor)
    result = ts_extract.extract_ts_like("export const a = 1;\n", "web/app.ts", "typescript")
    assert result.status == call_graph.COVERAGE_PARSE_PARTIAL, "unknown error state must not be INDEXED"


@pytest.mark.skipif(not ts_extract.is_tree_sitter_functional(), reason="tree-sitter unavailable")
def test_has_error_probe_failure_marks_extract_parse_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RootProxy:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> object:
            if name == "has_error":
                raise RuntimeError("forced: binding quirk")
            return getattr(self._inner, name)

    real_parse = ts_extract._parse

    def _parse(parser, data):
        tree = real_parse(parser, data)
        return SimpleNamespace(root_node=_RootProxy(tree.root_node))

    source = "export function f() { return g(); }\nfunction g() { return 1; }\n"
    clean = ts_extract.extract_ts_like(source, "web/app.ts", "typescript")
    assert clean.status == call_graph.COVERAGE_INDEXED, "control: a clean parse is INDEXED"

    monkeypatch.setattr(ts_extract, "_parse", _parse)
    result = ts_extract.extract_ts_like(source, "web/app.ts", "typescript")
    assert result.status == call_graph.COVERAGE_PARSE_PARTIAL, "unknown error state must not be INDEXED"


# T8-[kernel_repo_intel | _not_found | except Exception: | 1]
# suggestions are advisory; the NOT_FOUND status stays typed.
def test_not_found_survives_suggestion_failure() -> None:
    class _Graph:
        def search_names(self, symbol: str, limit: int = 5):
            raise RuntimeError("forced: search")

        def coverage_summary(self):
            return {}

    result = repo_intel._not_found("explore", "Missing", _Graph())
    assert result["status"] == "NOT_FOUND"
    assert result["suggestions"] == []


# T8-[kernel_repo_intel | _ensure_focus_entry | except Exception: | 1]
# focused orientation is documented fail-open; the map is returned unchanged.
def test_focus_entry_failure_is_fail_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _boom(file_path: Path, repo_root: Path):
        raise RuntimeError("forced: file entry")

    monkeypatch.setattr(repo_intel._repo_map_module, "_file_entry", _boom)
    assert repo_intel._ensure_focus_entry(tmp_path, "MAP", "pkg/mod.py") == "MAP"


# T8-[kernel_repo_catalog | build_structural_index | except Exception as exc: | 1]
# a failed outline is recorded on the file's outline as degraded with a reason.
def test_outline_failure_is_typed_degraded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    def _boom(source: str, relpath: str):
        raise RuntimeError("forced: outline")

    monkeypatch.setattr(catalog, "outline_source", _boom)
    index = catalog.build_structural_index(str(tmp_path))
    modules = [m for m in index.modules if m.path == "mod.py"]
    assert modules, "the file must still be indexed"
    assert modules[0].degraded is True
    assert modules[0].parser == "unavailable"


# T8-[repo_map | _file_entry_with_mode | except Exception as exc: | 1]
# outline failure on a non-scanned language lists the file as text scan and keeps the reason.
def test_file_entry_outline_failure_keeps_text_scan_and_reason(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "svc.go"
    src.write_text("package svc\n\nfunc Run() {}\n", encoding="utf-8")

    def _boom(source: str, rel_path: str):
        raise RuntimeError("forced: outline sigs")

    monkeypatch.setattr(repo_map, "_outline_sigs", _boom)
    monkeypatch.setattr(repo_map, "_TREE_SITTER_AVAILABLE", True)
    entry, mode, failure = repo_map._file_entry_with_mode(src, tmp_path)
    assert mode == repo_map._TEXT_SCAN
    assert failure == "RuntimeError"
    assert entry is not None and "svc.go" in entry


# T8-[repo_map | _file_entry_with_mode | except Exception as exc: | 2]
# a tree-sitter extract failure degrades that file to the text scan and records the failure type.
def test_file_entry_tree_sitter_failure_degrades_and_records(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "mod.py"
    src.write_text("def handler(x):\n    return x\n", encoding="utf-8")

    def _boom(source_bytes: bytes):
        raise RuntimeError("forced: python extract")

    monkeypatch.setattr(repo_map, "_TREE_SITTER_AVAILABLE", True)
    monkeypatch.setattr(repo_map, "_get_parser", lambda *a, **k: object())
    monkeypatch.setattr(repo_map, "_extract_python_sigs", _boom)
    entry, mode, failure = repo_map._file_entry_with_mode(src, tmp_path)
    assert failure == "RuntimeError"
    assert mode == repo_map._TEXT_SCAN
    assert entry is not None


# T8-[ts_dispatch | _vue_outline | except Exception as exc: | 1]
# a broken <script> block degrades the SFC outline with a typed reason.
def test_vue_script_parse_failure_is_typed_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "<script>\nexport const a = 1\n</script>\n<template><div>{{ a }}</div></template>\n"

    def _boom(language: str, text: str):
        raise RuntimeError("forced: parse")

    monkeypatch.setattr(ts_dispatch, "_parse_source", _boom)
    outline = ts_dispatch._vue_outline(source, "Widget.vue", "extension")
    assert outline["degraded"] is True
    assert "script:RuntimeError" in str(outline.get("degraded_reason", ""))


# T8-[ts_dispatch | structural_spans | except Exception as exc: | 1]
# a script-block parser failure in structural spans is recorded as a parser-status span.
def test_structural_spans_script_failure_records_parser_status(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "<script>\nexport const a = 1\n</script>\n"

    def _boom(language: str, text: str):
        raise RuntimeError("forced: parse")

    monkeypatch.setattr(ts_dispatch, "_parse_source", _boom)
    result = ts_dispatch.structural_spans(source, "Widget.vue")
    statuses = [span for span in result["spans"] if span.get("kind") == "parser-status"]
    assert statuses and statuses[0]["reason"] == "RuntimeError"
    assert result["degraded"] is True


# T8-[ts_dispatch | structural_spans | except Exception as exc: | 2]
# an unavailable parser refuses structural edits with a typed unavailable outcome.
def test_structural_spans_unavailable_parser_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(language: str, text: str):
        raise RuntimeError("forced: parser missing")

    monkeypatch.setattr(ts_dispatch, "_parse_source", _boom)
    result = ts_dispatch.structural_spans("export const a = 1;\n", "app.ts")
    assert result["parser"] == "unavailable"
    assert result["degraded"] is True
    assert result["reason"] == "RuntimeError"


# T8-[ts_dispatch | _python_outline_parser_identity | except Exception: | 1]
# a Python grammar probe failure reports the stdlib ast identity, never a tree-sitter one.
def test_python_parser_identity_falls_back_to_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("tree_sitter_language_pack")

    def _get_parser(name: str):
        raise RuntimeError("forced: grammar missing")

    fake.get_parser = _get_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    assert ts_dispatch._python_outline_parser_identity() == ("ast", "AVAILABLE", "stdlib")


# T8-[ts_dispatch | outline_capability_key | except Exception as exc: | 1]
# a Vue grammar probe failure is reported in the outline capability key.
def test_vue_capability_probe_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> bool:
        raise RuntimeError("forced: probe")

    monkeypatch.setattr(ts_dispatch, "_vue_grammar_available", _boom)
    key = ts_dispatch.outline_capability_key("Widget.vue", "<template/>")
    assert ":DEGRADED:" in key


# T8-[ts_dispatch | outline_capability_key | except Exception as exc: | 2]
# a TS grammar lookup failure is reported in the outline capability key.
def test_ts_capability_probe_failure_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("tree_sitter_language_pack")

    def _get_parser(name: str):
        raise RuntimeError("forced: grammar missing")

    fake.get_parser = _get_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)
    key = ts_dispatch.outline_capability_key("app.ts", "export const a = 1;\n")
    assert key.startswith("typescript:DEGRADED:RuntimeError") or ":DEGRADED:RuntimeError:" in key


# T8-[ts_dispatch | outline_file | except Exception as exc: | 1]
# an unreadable file yields a lexical outline that names the read failure.
def test_outline_file_unreadable_is_typed_lexical(tmp_path: Path) -> None:
    result = ts_dispatch.outline_file(tmp_path / "missing.ts")
    assert result["degraded"] is True
    assert "FileNotFoundError" in str(result)


# ── Inventory guard and tag self-check ────────────────────────────────────────────────────────────

# Unit sites: 24 silent sites (verdicts in the T8H-REPOINTEL md).
_UNIT_SITES: set[ExceptSite] = {
    ("kernel_repo_catalog", "_graph_dependencies", "except Exception as exc:", 2),
    ("kernel_repo_catalog", "build_structural_index", "except Exception as exc:", 1),
    ("kernel_repo_intel", "_not_found", "except Exception:", 1),
    ("kernel_repo_intel", "_ensure_focus_entry", "except Exception:", 1),
    ("call_graph", "_expr_src", "except Exception:", 1),
    ("call_graph", "dispatch_boundaries_in_file", "except Exception as exc:", 1),
    ("call_graph", "<module>", "except Exception:", 1),
    ("lsp_resolve", "<module>", "except Exception:", 1),
    ("lsp_resolve", "_to_target", "except Exception:", 1),
    ("repo_map", "_file_entry_with_mode", "except Exception as exc:", 1),
    ("repo_map", "_file_entry_with_mode", "except Exception as exc:", 2),
    ("repo_map", "_record_commit_sha_lookup_failure", "except BaseException:", 1),
    ("repo_map", "_record_commit_sha_lookup_failure", "except BaseException:", 2),
    ("repo_map", "_record_commit_sha_lookup_failure", "except Exception:", 1),
    ("repo_map", "_record_commit_sha_lookup_failure", "except Exception:", 2),
    ("ts_dispatch", "_vue_grammar_available", "except Exception:", 1),
    ("ts_dispatch", "_vue_outline", "except Exception as exc:", 1),
    ("ts_dispatch", "structural_spans", "except Exception as exc:", 1),
    ("ts_dispatch", "structural_spans", "except Exception as exc:", 2),
    ("ts_dispatch", "_python_outline_parser_identity", "except Exception:", 1),
    ("ts_dispatch", "outline_capability_key", "except Exception as exc:", 1),
    ("ts_dispatch", "outline_capability_key", "except Exception as exc:", 2),
    ("ts_dispatch", "outline_file", "except Exception as exc:", 1),
    ("ts_extract", "_has_error", "except Exception as exc:", 1),
}

# The remaining 27 are census-v2 LOUD (census counts any logger.* call as loud,
# so several are debug-only; flagged in the md for re-triage).
_CENSUS_LOUD: set[ExceptSite] = {
    ("kernel_repo_catalog", "_graph_dependencies", "except Exception as exc:", 1),
    ("kernel_repo_intel", "_journal_repo_map_receipt", "except Exception:", 1),
    ("call_graph", "_atomic_write_text", "except BaseException:", 1),
    ("call_graph", "CodeGraph._apply_lsp_resolution", "except Exception as exc:", 1),
    ("call_graph", "CodeGraph._apply_lsp_resolution", "except Exception as exc:", 2),
    ("call_graph", "CodeGraph.pagerank", "except Exception as exc:", 1),
    ("call_graph", "repo_cache_key", "except Exception as exc:", 1),
    ("call_graph", "_copy_shared_sidecar", "except Exception as exc:", 1),
    ("call_graph", "warm_cache_from_shared", "except Exception as exc:", 1),
    ("call_graph", "publish_cache_to_shared", "except Exception as exc:", 1),
    ("call_graph", "publish_cache_to_shared", "except Exception as exc:", 2),
    ("call_graph", "_tree_fingerprint", "except Exception as exc:", 1),
    ("lsp_resolve", "_resolve_sites_blocking", "except Exception as exc:", 1),
    ("lsp_resolve", "resolve_file_full.resolve", "except BaseException as error:", 1),
    ("lsp_resolve", "resolve_file_full", "except Exception as exc:", 1),
    ("lsp_resolve", "resolve_file_full", "except Exception as exc:", 2),
    ("repo_map", "build_repo_map_with_stats", "except Exception as exc:", 1),
    ("ts_dispatch", "_scan_ts", "except Exception as exc:", 1),
    ("ts_dispatch", "dispatch_boundaries_in_file", "except Exception as exc:", 1),
    ("ts_dispatch", "_python_ast_fallback", "except Exception as exc:", 1),
    ("ts_dispatch", "_structure_outline", "except Exception as exc:", 1),
    ("ts_dispatch", "outline_source", "except Exception as exc:", 1),
    ("ts_extract", "<module>", "except Exception as _exc:", 1),
    ("ts_extract", "is_tree_sitter_functional", "except Exception as exc:", 1),
    ("ts_extract", "_TSVisitor.run", "except Exception as exc:", 1),
    ("ts_extract", "extract_ts_like", "except Exception as exc:", 1),
    ("ts_extract", "extract_ts_like", "except Exception as exc:", 2),
}


def test_repointel_broad_excepts_are_triaged() -> None:
    assert_triaged(_broad_except_sites(_MODULE_FILES), _UNIT_SITES | _CENSUS_LOUD)


def test_repointel_journal_tags_point_at_broad_except_sites() -> None:
    assert_site_tags(Path(__file__), _broad_except_sites(_MODULE_FILES))


def test_repointel_inventory_survives_line_shift(tmp_path: Path) -> None:
    found = _broad_except_sites(shifted_module_paths(_MODULE_FILES, tmp_path))
    assert_triaged(found, _UNIT_SITES | _CENSUS_LOUD)
    assert_site_tags(Path(__file__), found)


def test_repointel_inventory_rejects_missing_site(tmp_path: Path) -> None:
    paths = edited_module_paths(
        _MODULE_FILES, tmp_path, "ts_extract",
        "def _has_error(", "def _renamed_has_error(",
    )
    stale = ("ts_extract", "_has_error", "except Exception as exc:", 1)
    found = _broad_except_sites(paths)
    with pytest.raises(AssertionError) as failure:
        assert_triaged(found, _UNIT_SITES | _CENSUS_LOUD)
    assert f"stale: [{stale!r}]" in str(failure.value)
    with pytest.raises(AssertionError) as tag_failure:
        assert_site_tags(Path(__file__), found)
    assert "ts_extract | _has_error | except Exception as exc: | 1" in str(tag_failure.value)


def test_repointel_inventory_rejects_untriaged_site(tmp_path: Path) -> None:
    paths = edited_module_paths(_MODULE_FILES, tmp_path, "ts_extract", None, UNREVIEWED_HANDLER)
    untriaged = ("ts_extract", "_t8_unreviewed", "except Exception:", 1)
    with pytest.raises(AssertionError) as failure:
        assert_triaged(_broad_except_sites(paths), _UNIT_SITES | _CENSUS_LOUD)
    assert f"untriaged: [{untriaged!r}]; stale: []" in str(failure.value)
