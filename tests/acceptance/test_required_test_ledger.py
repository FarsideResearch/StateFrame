"""Scripted public run for the opt-in host required-test ledger."""
from __future__ import annotations

import json

import pytest

from ._support import run_case


pytestmark = pytest.mark.host_sandbox


def test_scripted_run_reports_baseline_test_failure_once(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "app.py").write_text("def value():\n    return 'good'\n", encoding="utf-8")
    (repo / "tests" / "test_value.py").write_text(
        "from app import value\n\n\ndef test_value_is_good():\n    assert value() == 'good'\n",
        encoding="utf-8",
    )
    script = tmp_path / "required-test-ledger.script.json"
    script.write_text(json.dumps({
        "model": "scripted",
        "tokenize": {"mode": "ok"},
        "steps": [
            {"tool": "read", "arguments": {"path": "app.py"}},
            {"tool": "str_replace", "arguments": {
                "path": "app.py",
                "old_str": "return 'good'",
                "new_str": "return 'bad'",
            }},
            {"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
            {"tool": "done", "arguments": {"summary": "The requested edit is complete."}},
        ],
    }), encoding="utf-8")

    case = run_case(
        tmp_path,
        str(script),
        repo=repo,
        default_test_cmd=False,
        flags=("--test-cmd", "python -m pytest -q tests/test_value.py", "--target", "app.py"),
        env_overrides={"STATEFRAME_REQUIRED_TEST_LEDGER": "on"},
        task="Change app.py and run its selected pytest check.",
    )

    assert case.summary is not None, case.process.stdout + case.process.stderr
    ledger = case.summary["required_test_ledger"]
    assert case.summary["tests_before"]["passed"] is True
    assert case.summary["tests_after"]["passed"] is False
    assert ledger["state"] == "regressed"
    assert ledger["missing"] == []
    assert ledger["failed"] == ["tests/test_value.py::test_value_is_good"]
    assert ledger["passed"] == []
    fire_count = int(ledger["state"] == "regressed")
    assert fire_count == 1
