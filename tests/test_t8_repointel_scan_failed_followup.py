"""Scan-failed dispatch markers survive localization and the TS dispatch scanner.

A failed boundary scan returns a file-level ``scan_failed`` marker meaning "dynamic dispatch here is
unknown". Two places used to lose it: localization dropped it whenever task terms were set (its note
never names the task's terms), and the TS/JS scanner turned its own scan error into an empty list.
"""
from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_localize as localize
from stateframe.engine.repo_intel_core import call_graph, ts_dispatch


def _inventory(path: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(files=[SimpleNamespace(path=path, text=text)])


def test_localize_keeps_scan_failed_marker_when_task_terms_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # Localization imports the vendor package by its top-level name (the engine dir is on sys.path).
    graph_module = importlib.import_module("repo_intel_core.call_graph")
    marker = graph_module._scan_failed_boundary("pkg/app.ts", "RuntimeError")
    unrelated = graph_module.DispatchBoundary(file="pkg/app.ts", lineno=7, kind="getattr",
                                              key_expr="handler", note="registry lookup")
    monkeypatch.setattr(graph_module, "dispatch_boundaries_in_file", lambda _text, _path: [marker, unrelated])
    monkeypatch.setattr(localize, "_symbol_for_line", lambda *_args: None)
    monkeypatch.setattr(localize, "_source_ref", lambda _source, line, why: {"line": line, "why": why})
    monkeypatch.setattr(localize, "_candidate", lambda **fields: fields)
    rows = localize._dynamic_bindings(_inventory("pkg/app.ts", "x"), object(), {"terms": ("checkout",)})
    assert [row["binding_kind"] for row in rows] == [call_graph.SCAN_FAILED_KIND]


def test_ts_scanner_error_returns_scan_failed_marker(monkeypatch: pytest.MonkeyPatch,
                                                    caplog: pytest.LogCaptureFixture) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("forced scan failure")

    monkeypatch.setattr(ts_dispatch, "_scan_ts", boom)
    with caplog.at_level(logging.WARNING):
        boundaries = ts_dispatch.dispatch_boundaries_in_file("const a = 1;", "web/app.ts")
    assert [(b.kind, b.lineno) for b in boundaries] == [(call_graph.SCAN_FAILED_KIND, 1)]
    assert "RuntimeError" in boundaries[0].note
    assert any("web/app.ts" in record.getMessage() for record in caplog.records
               if record.levelno >= logging.WARNING)


def test_ts_scanner_success_is_unchanged() -> None:
    assert all(b.kind != call_graph.SCAN_FAILED_KIND
               for b in ts_dispatch.dispatch_boundaries_in_file("const a = 1;", "web/app.ts"))
