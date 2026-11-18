"""Focused tests for checkpoint wait diagnostics in process fixtures."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fixtures.process_signals import signal_run_owned, wait_for_checkpoint_saved


def _write_events(run_dir: Path, *events: dict[str, object]) -> Path:
    journal = run_dir / "checkpoints" / "events.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_bytes(b"".join(
        json.dumps(event).encode("utf-8") + b"\n" for event in events
    ))
    return journal


def test_checkpoint_failure_fails_fast_with_engine_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_events(run_dir, {"event": "checkpoint_failed", "failure_reason": "write_failed"})
    started = time.monotonic()

    with pytest.raises(AssertionError) as error:
        wait_for_checkpoint_saved(run_dir, deadline=started + 3, what="failed save")

    assert time.monotonic() - started < 1
    assert "checkpoint_failed" in str(error.value)
    assert "failure_reason" in str(error.value)
    assert "write_failed" in str(error.value)


def test_disabled_checkpoint_saving_fails_fast_with_engine_reason(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_events(
        run_dir,
        {"event": "checkpoint_disabled", "reason": "three failed saves"},
    )
    started = time.monotonic()

    with pytest.raises(AssertionError) as error:
        wait_for_checkpoint_saved(run_dir, deadline=started + 3, what="disabled save")

    assert time.monotonic() - started < 1
    assert "checkpoint_disabled" in str(error.value)
    assert "three failed saves" in str(error.value)


def test_exited_child_fails_fast_with_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(23)"])
    assert child.wait(timeout=5) == 23
    started = time.monotonic()

    with pytest.raises(AssertionError) as error:
        signal_run_owned(
            run_dir, child, signal.SIGKILL, deadline=started + 10, what="exited child",
        )

    assert time.monotonic() - started < 0.5
    assert "exit code 23" in str(error.value)


def test_successful_save_returns(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_events(run_dir, {"event": "checkpoint_saved"})

    assert wait_for_checkpoint_saved(
        run_dir, deadline=time.monotonic() + 1, what="saved run",
    ) is None


def test_incomplete_trailing_save_record_is_ignored(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    journal = run_dir / "checkpoints" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_bytes(b'{"event":"checkpoint_saved"')

    with pytest.raises(AssertionError, match="saw 0/1 saves"):
        wait_for_checkpoint_saved(
            run_dir, deadline=time.monotonic() - 1, what="incomplete save",
        )
