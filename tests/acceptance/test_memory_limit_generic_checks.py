"""Generic Go and Node checks keep working under Linux memory limits."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ._support import run_case


_node = shutil.which("node")
_node_in_sandbox_mounts = bool(_node and Path(_node).resolve().is_relative_to("/usr"))
pytestmark = pytest.mark.host_sandbox


def _script(path: str, old: str, new: str):
    return json.dumps({
        "model": "scripted",
        "tokenize": {"mode": "ok"},
        "steps": [
            {"tool": "read", "arguments": {"path": path}},
            {"tool": "run_tests", "arguments": {"target": ""}},
            {"tool": "str_replace", "arguments": {
                "path": path, "old_str": old, "new_str": new}},
            {"tool": "run_tests", "arguments": {"target": ""}},
            {"tool": "done", "arguments": {"summary": "Fixed the check and verified it."}},
        ],
    })


def _assert_before_after(case):
    assert case.summary is not None, case.process.stdout + case.process.stderr
    assert case.summary["tests_before"]["passed"] is False, case.summary
    assert case.summary["tests_after"]["passed"] is True, case.summary
    assert case.summary["succeeded"] is True, case.summary
    case.assert_reply("check exit 0")


@pytest.mark.skipif(shutil.which("go") is None, reason="go is not installed")
def test_go_generic_check_reserves_virtual_address_space(tmp_path, monkeypatch):
    monkeypatch.setenv("GOPROXY", "off")
    monkeypatch.setenv("GOTOOLCHAIN", "local")
    repo = tmp_path / "go-repo"
    repo.mkdir()
    (repo / "go.mod").write_text("module example.com/memorycheck\n\ngo 1.21\n")
    (repo / "app.go").write_text('package memorycheck\n\nfunc Value() string { return "bad" }\n')
    (repo / "app_test.go").write_text(
        'package memorycheck\n\nimport "testing"\n\n'
        'func TestValue(t *testing.T) { if Value() != "good" { t.Fatal("bad value") } }\n')
    script = tmp_path / "go-check.script.json"
    script.write_text(_script("app.go", 'return "bad"', 'return "good"'))
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "go test ./...", "--target", "app.go"), timeout=360)
    _assert_before_after(case)


@pytest.mark.skipif(not _node, reason="node is not installed")
@pytest.mark.skipif(not _node_in_sandbox_mounts,
                    reason="node is outside the read-only runtime roots mounted into checks")
def test_node_generic_check_reserves_virtual_address_space(tmp_path):
    repo = tmp_path / "node-repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.js").write_text("module.exports = { value: 'bad' };\n")
    (repo / "tests" / "app.test.js").write_text(
        "const test = require('node:test');\n"
        "const assert = require('node:assert/strict');\n"
        "test('value', () => assert.equal(require('../app').value, 'good'));\n")
    script = tmp_path / "node-check.script.json"
    script.write_text(_script("app.js", "value: 'bad'", "value: 'good'"))
    case = run_case(tmp_path, str(script), repo=repo,
                    flags=("--test-cmd", "node --test", "--target", "app.js"))
    _assert_before_after(case)
