"""Review regressions for exact in-place promotion."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

from stateframe import workcopy
from tests.test_delivery_refusal_not_submitted import _execute_prepared, _prepare_execution


def test_conflict_after_first_replace_reports_the_partial_apply(
        tmp_path: Path, monkeypatch) -> None:
    options, run, _work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch,
        {"sample.py": b"value = 2\n", "created.py": b"created\n"})
    spec["public_row"]["task_file_scope"] = None

    real_replace = os.replace
    real_apply = workcopy.apply_in_place
    applying = False
    raced = False

    def replace_then_edit_next_original(src, dst, *args, **kwargs):
        nonlocal raced
        result = real_replace(src, dst, *args, **kwargs)
        if applying and not raced and Path(dst).name == "created.py":
            # `created.py` sorts before `sample.py` in the apply transaction.
            # This race starts after publication; the earlier replacement must be reported.
            (repo / "sample.py").write_bytes(b"value = concurrently edited\n")
            raced = True
        return result

    def arm_apply(target, **kwargs):
        nonlocal applying
        applying = True
        return real_apply(target, **kwargs)

    monkeypatch.setattr(workcopy.os, "replace", replace_then_edit_next_original)
    monkeypatch.setattr(workcopy, "apply_in_place", arm_apply)
    summary = _execute_prepared(options, run, _work, spec)

    assert raced
    assert summary.get("apply_error"), "the injected original drift should use apply_error"
    assert summary["exit_code"] == 2
    assert "already applied: ['created.py']" in summary["apply_error"]
    assert (repo / "sample.py").read_bytes() == b"value = concurrently edited\n"
    assert (repo / "created.py").read_bytes() == b"created\n"


def test_conflict_during_staging_writes_no_destinations(tmp_path: Path, monkeypatch) -> None:
    options, run, _work, spec, repo = _prepare_execution(
        tmp_path, monkeypatch,
        {"sample.py": b"value = 2\n", "created.py": b"created\n"})
    spec["public_row"]["task_file_scope"] = None

    real_open = os.open
    real_apply = workcopy.apply_in_place
    applying = raced = False

    def stage_then_edit_next_original(path, flags, *args, **kwargs):
        nonlocal raced
        descriptor = real_open(path, flags, *args, **kwargs)
        if applying and not raced and Path(path).name.startswith(".stateframe-"):
            (repo / "sample.py").write_bytes(b"value = concurrently edited\n")
            raced = True
        return descriptor

    def arm_apply(target, **kwargs):
        nonlocal applying
        applying = True
        return real_apply(target, **kwargs)

    monkeypatch.setattr(workcopy.os, "open", stage_then_edit_next_original)
    monkeypatch.setattr(workcopy, "apply_in_place", arm_apply)
    summary = _execute_prepared(options, run, _work, spec)

    assert raced
    assert summary.get("apply_error")
    assert (repo / "sample.py").read_bytes() == b"value = concurrently edited\n"
    assert not (repo / "created.py").exists()
    assert not list(repo.glob(".stateframe-*.tmp"))


@pytest.mark.parametrize("failure", [errno.EIO, errno.ENOSPC, "interrupt"])
def test_staging_failure_leaves_destinations_and_no_temps(tmp_path: Path, monkeypatch, failure) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    before = {"a.py": b"old a\n", "b.py": b"old b\n"}
    for rel, data in before.items():
        (repo / rel).write_bytes(data)
    changes = {"a.py": b"new a\n", "b.py": b"new b\n"}
    baseline = {rel: hashlib.sha256(data).hexdigest() for rel, data in before.items()}
    real_write = os.write
    writes = 0

    def fail_during_staging(descriptor, data):
        nonlocal writes
        writes += 1
        if writes == 2:
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise OSError(failure, "injected staging failure")
        return real_write(descriptor, data)

    monkeypatch.setattr(workcopy.os, "write", fail_during_staging)
    expected = KeyboardInterrupt if failure == "interrupt" else workcopy.WorkCopyError
    with pytest.raises(expected):
        workcopy.guarded_apply(repo, changes, baseline, what="test apply")

    assert {rel: (repo / rel).read_bytes() for rel in before} == before
    assert not list(repo.glob(".stateframe-*.tmp"))
