"""The capability receipt never reports the repo map from an unaudited grammar bundle.

The product only uses its audited per-language grammar packages, installed under the language-pack
module name by ``ts_grammars.install_as_language_pack``. When that install fails, the receipt must
say so and report the repo map as unavailable, instead of falling back to whatever bundled
``tree_sitter_language_pack`` happens to be importable.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types

import pytest

from stateframe import runner


def _run_probe(monkeypatch: pytest.MonkeyPatch, *, install) -> dict:
    grammars = types.ModuleType("stateframe.ts_grammars")
    grammars.install_as_language_pack = install
    grammars.available_languages = lambda: ["python"]
    grammars.get_parser = lambda _name: types.SimpleNamespace(parse=lambda _data: None)
    bundle = types.ModuleType("tree_sitter_language_pack")
    bundle.__version__ = "9.9.9"
    bundle.get_parser = lambda _name: None
    monkeypatch.setitem(sys.modules, "stateframe.ts_grammars", grammars)
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", bundle)
    import stateframe
    monkeypatch.setattr(stateframe, "ts_grammars", grammars, raising=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(runner._CAPABILITY_PROBE, "<capability-probe>", "exec"), {"__name__": "__main__"})
    return json.loads(out.getvalue().strip().splitlines()[-1])["repo_map_ast"]


def test_failed_grammar_install_is_recorded_and_never_reports_the_bundle(monkeypatch):
    def broken():
        raise RuntimeError("forced install failure")

    repo_map = _run_probe(monkeypatch, install=broken)
    assert repo_map["enabled"] is False
    assert "grammar install failed: RuntimeError" in repo_map["detail"]
    assert "9.9.9" not in repo_map["detail"]
    assert "forced install failure" not in repo_map["detail"]  # type only, never the message


def test_working_grammar_install_reports_its_languages(monkeypatch):
    repo_map = _run_probe(monkeypatch, install=lambda: True)
    assert repo_map == {"enabled": True, "detail": "1 languages: python"}
