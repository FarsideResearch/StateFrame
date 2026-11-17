"""MutationExecutor must survive a checkpoint pickle round trip with its launch ledger.

Regression: the ledger was opened with os.fdopen(fd), so its name was an int fd and the
checkpoint pickler refused it ("unsupported stream FileIO name=<fd>"). Mutation runs could
never checkpoint, which left crash-resume broken for a shipped feature.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import kernel_checkpoint
from kernel_mutate_exec import MutationExecutor

ENGINE = Path(kernel_checkpoint.__file__).resolve().parent
_CTX = {"framecells": {}, "framefuncs": {}, "cvars": {}, "hostfns": {}, "participants": {}}


def _spec(launch: str) -> SimpleNamespace:
    return SimpleNamespace(launch=launch, call="c", role="payload")


def _rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "launches.jsonl").read_text().splitlines()]


def _executor(tmp_path: Path) -> MutationExecutor:
    root = tmp_path / ("a" * 32)
    root.mkdir()
    return MutationExecutor(root, network_envelope="none")


def _load(blob: bytes):
    return kernel_checkpoint._Unpickler(io.BytesIO(blob), {"framecells": {}, "framefuncs": {}}).load()


def test_executor_pickles_through_the_checkpoint_pickler(tmp_path):
    executor = _executor(tmp_path)
    try:
        executor._record(_spec("l1"), "PREPARED")
        blob = kernel_checkpoint.dumps({"executor": executor}, _CTX)
        assert blob
    finally:
        executor.close()


def test_restored_ledger_appends_after_existing_rows(tmp_path):
    executor = _executor(tmp_path)
    try:
        executor._record(_spec("l1"), "PREPARED")
        blob = kernel_checkpoint.dumps({"executor": executor}, _CTX)
        restored = _load(blob)["executor"]
        restored._record(_spec("l2"), "PREPARED")
        assert [row["launch"] for row in _rows(executor.root)] == ["l1", "l2"]
        restored._ledger.close()
    finally:
        executor.close()


def test_restored_ledger_is_durable_before_the_hook_runs(tmp_path):
    # The hook contract: a stage hook sees the row already on disk (fsync after flush).
    executor = _executor(tmp_path)
    try:
        blob = kernel_checkpoint.dumps({"executor": executor}, _CTX)
        restored = _load(blob)["executor"]
        seen = []
        restored._launch_stage_hook = lambda stage: seen.append(_rows(executor.root)[-1]["state"])
        restored._record(_spec("l1"), "LAUNCHED")
        assert seen == ["LAUNCHED"]
        restored._ledger.close()
    finally:
        executor.close()


def test_ledger_keeps_private_mode_and_exclusive_create(tmp_path):
    executor = _executor(tmp_path)
    try:
        ledger = executor.root / "launches.jsonl"
        assert ledger.stat().st_mode & 0o777 == 0o600
        assert executor._ledger.name == str(ledger)
    finally:
        executor.close()


def test_resume_in_a_fresh_process_appends_to_the_ledger(tmp_path):
    executor = _executor(tmp_path)
    try:
        executor._record(_spec("l1"), "PREPARED")
        blob_path = tmp_path / "ckpt.bin"
        blob_path.write_bytes(kernel_checkpoint.dumps({"executor": executor}, _CTX))
        code = (
            "import io, sys\n"
            "from types import SimpleNamespace\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import kernel_checkpoint\n"
            "blob = open(sys.argv[2], 'rb').read()\n"
            "ex = kernel_checkpoint._Unpickler(io.BytesIO(blob), {'framecells': {}, 'framefuncs': {}}).load()['executor']\n"
            "ex._record(SimpleNamespace(launch='l2', call='c', role='payload'), 'PREPARED')\n"
            "ex._ledger.close()\n"
        )
        result = subprocess.run([sys.executable, "-c", code, str(ENGINE), str(blob_path)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert [row["launch"] for row in _rows(executor.root)] == ["l1", "l2"]
    finally:
        executor.close()
