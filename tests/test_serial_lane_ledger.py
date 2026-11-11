"""Unit tests for writing and checking serial-lane session ledgers."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from _timeouts import scaled
import serial_lane_plugin


def _run_checker(ledger: Path) -> subprocess.CompletedProcess[str]:
    tests_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(tests_dir), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-m", "serial_lane_ledger", str(ledger)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=scaled(10),
    )


def test_ledger_writer_appends_one_json_record_per_session(tmp_path):
    ledger = tmp_path / "serial-lane.jsonl"

    serial_lane_plugin._append_serial_lane_ledger(
        ledger, collected=2, ran=0, deferred=2
    )
    serial_lane_plugin._append_serial_lane_ledger(
        ledger, collected=2, ran=2, deferred=0
    )

    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"collected": 2, "deferred": 2, "ran": 0},
        {"collected": 2, "deferred": 0, "ran": 2},
    ]


def test_checker_passes_when_serial_step_ran_cases(tmp_path):
    ledger = tmp_path / "serial-lane.jsonl"
    ledger.write_text(
        '{"collected":2,"ran":0,"deferred":2}\n'
        '{"collected":2,"ran":2,"deferred":0}\n',
        encoding="utf-8",
    )

    result = _run_checker(ledger)

    assert result.returncode == 0, result.stderr
    assert "2 ran" in result.stdout


def test_planted_all_deferred_control_fails_checker(tmp_path):
    ledger = tmp_path / "planted-control.jsonl"
    ledger.write_text(
        '{"collected":2,"ran":0,"deferred":2}\n'
        '{"collected":2,"ran":0,"deferred":2}\n'
        '{"collected":2,"ran":0,"deferred":2}\n',
        encoding="utf-8",
    )

    result = _run_checker(ledger)

    assert result.returncode == 1
    assert "6 corpus cases collected, 0 ran" in result.stderr


def test_empty_or_absent_ledger_passes_with_zero_collected(tmp_path):
    absent = _run_checker(tmp_path / "absent.jsonl")
    empty_path = tmp_path / "empty.jsonl"
    empty_path.touch()
    empty = _run_checker(empty_path)

    assert absent.returncode == 0, absent.stderr
    assert empty.returncode == 0, empty.stderr
    assert "no corpus cases collected" in absent.stdout
    assert "no corpus cases collected" in empty.stdout


def test_checker_rejects_malformed_ledger_record(tmp_path):
    ledger = tmp_path / "malformed.jsonl"
    ledger.write_text('{"collected":1,"ran":0}\n', encoding="utf-8")

    result = _run_checker(ledger)

    assert result.returncode == 2
    assert "must contain exactly" in result.stderr
