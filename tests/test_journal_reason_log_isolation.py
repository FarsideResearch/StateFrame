"""Regression coverage for warning state leaking into journal assertions."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    "tests/test_request_determinism.py::test_wire_ids_rederive_from_disk_replayed_action_receipt_history",
    "tests/test_wave4_startup_journal.py::test_event_bytes_and_receipt_references_match_through_resume",
)


def test_named_journal_cases_pass_with_warning_before_each_test(tmp_path):
    environment = os.environ.copy()
    environment["TEST_WARN_REASON_PLUGIN"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "tests"), environment.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-n", "0", "-p", "_journal_reason_log",
         "--basetemp", str(tmp_path / "warning-plugin-child"), *TARGETS],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, f"warning-plugin run returned {result.returncode}"
    warnings = (result.stdout + result.stderr).count("WARN test_warning_plugin")
    assert warnings == len(TARGETS), f"warning plugin emitted {warnings} times"
