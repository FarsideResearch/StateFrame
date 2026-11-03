"""A fixed finished run stays compact on disk with default retention, and stays inspectable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acceptance._support import run_case
from stateframe import inspect_run
from stateframe.request_records import list_request_records
from test_checkpoint_merge_regression import _script

# Measured 2026-11-03: live runs of discount_bug with a real run environment finished at 0.55 MB,
# where 0.3.16 kept 14.95 MB (env/ was removed only by this change; journals and preflight
# records were loose).
# The ceiling leaves headroom for small format changes, never for a dropped component.
FINISHED_RUN_BYTE_BUDGET = 1_500_000


def _apparent_bytes(root: Path) -> int:
    return sum(path.lstat().st_size for path in root.rglob("*")
               if path.is_file() and not path.is_symlink())


@pytest.mark.host_sandbox
def test_finished_run_with_env_stays_under_byte_budget_and_inspectable(
        tmp_path: Path, clean_test_environment) -> None:
    # A clean venv (pip + pytest, no editable installs) gives a run env StateFrame accepts; the
    # harness interpreter may carry editable paths outside the repo, which are refused.
    python = clean_test_environment(tmp_path / "test-venv")
    case = run_case(tmp_path, str(_script(tmp_path, "default")),
                    flags=("--run-env", "--test-python", str(python)))
    assert "run environment off" not in case.process.stdout, case.process.stdout
    assert case.process.returncode == 0, (case.process.stdout, case.process.stderr)
    assert case.summary and case.summary["succeeded"], case.summary
    run = case.run_dir

    assert not (run / "env").exists()
    assert json.loads((run / "env-manifest.json").read_text())["distributions"] is not None
    assert not any((run / "checkpoints").glob("*/state.pkl"))
    journals = list((run / "worker/measurement-runs").glob("*/events.jsonl*"))
    assert journals and all(path.name == "events.jsonl.xz" for path in journals)
    assert not any((run / "requests").glob("*.json"))
    assert (run / "requests.pack.xz").is_file()
    assert list_request_records(run) and len(list_request_records(run)) == len(case.requests)

    size = _apparent_bytes(run)
    assert size < FINISHED_RUN_BYTE_BUDGET, f"finished run uses {size} bytes"
    overview = inspect_run.overview(run)
    assert f"preflight records: {len(case.requests)} " in overview
    assert inspect_run.show_send(run, 1)
