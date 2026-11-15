"""ParallelSupervisor._materialize_submission: the WorkCopy baseline is built from per-file content hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import outcomes, runner, workcopy
from stateframe.engine.kernel_parallel import _snapshot
from stateframe.parallel_host import ParallelHostError, ParallelSupervisor


def _submitted(monkeypatch, candidate):
    monkeypatch.setattr(outcomes, "terminal_receipt", lambda run_dir: {"result": "SUBMITTED"})
    monkeypatch.setattr(outcomes, "sealed_artifact_digest", lambda terminal: "sha256:sealed")
    monkeypatch.setattr(runner, "final_candidate_with_file_digests",
                        lambda run_dir: (candidate, None))


def _snapshot_for(root: Path, files: dict[str, bytes]):
    source = root / "source"
    for relative, data in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return _snapshot(source, root / "parallel" / "call", 3)


def _worker(root: Path, snapshot):
    return SimpleNamespace(run_dir=root / "run", snapshot_path=snapshot.root,
                           snapshot_manifest_digest=snapshot.snapshot_manifest_digest)


def _spec(owns=("a.py",), creates=()):
    return SimpleNamespace(contract=SimpleNamespace(owns=owns, creates=creates))


def test_missing_owned_snapshot_file_is_a_host_error(tmp_path, monkeypatch):
    snapshot = _snapshot_for(tmp_path, {"b.py": b"outside scope\n"})
    _submitted(monkeypatch, {"a.py": b"new\n"})
    worker = _worker(tmp_path, snapshot)
    with pytest.raises(ParallelHostError, match="owned path is missing from its snapshot: a.py"):
        ParallelSupervisor._materialize_submission(worker, _spec(), tmp_path / "workspace")


def test_baseline_maps_each_owned_path_to_its_content_hash(tmp_path, monkeypatch):
    snapshot = _snapshot_for(tmp_path, {"a.py": b"old\n"})
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_bytes(b"old\n")
    _submitted(monkeypatch, {"a.py": b"new\n"})
    seen = {}
    real = workcopy.WorkCopy

    def capture(original, copy, baseline, **kwargs):
        seen.update(baseline)
        return real(original, copy, baseline, **kwargs)

    monkeypatch.setattr(workcopy, "WorkCopy", capture)
    worker = _worker(tmp_path, snapshot)
    ParallelSupervisor._materialize_submission(worker, _spec(), workspace)
    assert seen == {"a.py": hashlib.sha256(b"old\n").hexdigest()}
    assert (workspace / "a.py").read_bytes() == b"new\n"
