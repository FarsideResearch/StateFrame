"""Resume continues when optional archived checkpoint cleanup cannot finish."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from stateframe import resume


_REQUEST = b'{"model":"local","messages":[{"role":"user","content":"continue"}]}'
_RESPONSE = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'


def _prepare_run(tmp_path: Path, *, checkpoints: int = 1):
    run = tmp_path / "run"
    run.mkdir()
    spec = {"max_turns": 8, "environment": {}, "run_identity_seed": "resume-seed"}
    (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    wire = run / "wire"
    wire.mkdir()
    (wire / "0001.request.json").write_bytes(_REQUEST)
    (wire / "0001.response.json").write_bytes(_RESPONSE)
    (run / "sends.jsonl").write_text("", encoding="utf-8")

    root = run / "checkpoints"
    root.mkdir()
    snapshots = {}
    for number in range(1, checkpoints + 1):
        folder = root / f"{number:04d}-1"
        (folder / "run").mkdir(parents=True)
        state_bytes = f"state-{number}".encode()
        (folder / "state.pkl").write_bytes(state_bytes)
        (folder / "checkpoint.json").write_text(json.dumps({
            "schema": resume.CHECKPOINT_SCHEMA,
            "seq": number,
            "sends": 0,
            "state_bytes": len(state_bytes),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        }), encoding="utf-8")
        (folder / "run" / "snapshot.bin").write_bytes(f"snapshot-{number}".encode())
        snapshots[number] = folder

    for name in ("blob-pool", "journal-cache", ".partial.tmp"):
        payload = root / name
        payload.mkdir()
        (payload / "saved.bin").write_bytes(name.encode())
    (root / "LATEST").write_text("0001-1", encoding="utf-8")
    return run, spec, snapshots


def _archive(run: Path, spec: dict, *, attempt: int = 1) -> Path:
    return resume.archive_attempt({
        "run": run,
        "spec": spec,
        "attempt": attempt,
        "max_turns": 8,
        "previous_status": "interrupted",
    }, run.parent / "engine", use_checkpoint=False)


def _assert_resume_completed(run: Path, archive: Path) -> None:
    assert archive == run / "attempts" / "1"
    assert archive.is_dir()
    saved_spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    assert saved_spec["resume"]["attempt"] == 2


def _assert_one_diagnostic(capsys, run: Path, error_class: str, errno_name: str) -> None:
    captured = capsys.readouterr()
    lines = [line for line in (captured.out + captured.err).splitlines() if line]
    assert len(lines) == 1, lines
    assert error_class in lines[0]
    assert errno_name in lines[0]
    assert str(run) not in lines[0]


def test_read_only_checkpoint_directory_does_not_stop_resume(tmp_path, capsys):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("real permission failures cannot be exercised as root")

    run, spec, snapshots = _prepare_run(tmp_path)
    checkpoint = snapshots[1]
    state = checkpoint / "state.pkl"
    original = state.read_bytes()
    checkpoint.chmod(0o555)
    try:
        archive = _archive(run, spec)
        _assert_resume_completed(run, archive)
        assert (archive / "checkpoints" / checkpoint.name / "state.pkl").read_bytes() == original
        _assert_one_diagnostic(capsys, run, "PermissionError", "EACCES")
    finally:
        archived_checkpoint = run / "attempts" / "1" / "checkpoints" / checkpoint.name
        (archived_checkpoint if archived_checkpoint.exists() else checkpoint).chmod(0o755)


def test_vanished_checkpoint_entry_does_not_stop_resume(tmp_path, monkeypatch, capsys):
    run, spec, snapshots = _prepare_run(tmp_path)
    source_state = snapshots[1] / "state.pkl"
    state = run / "attempts" / "1" / "checkpoints" / snapshots[1].name / "state.pkl"
    original_unlink = Path.unlink
    error = FileNotFoundError(errno.ENOENT, "entry vanished", str(state))

    def vanish_then_report(path, *args, **kwargs):
        if path == state:
            original_unlink(path, *args, **kwargs)
            raise error
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", vanish_then_report)
    archive = _archive(run, spec)

    _assert_resume_completed(run, archive)
    assert not (archive / "checkpoints" / state.parent.name / "state.pkl").exists()
    _assert_one_diagnostic(capsys, run, "FileNotFoundError", "ENOENT")


def test_access_denied_one_of_three_checkpoints_is_skipped_and_can_be_retried(
    tmp_path, monkeypatch, capsys
):
    run, spec, snapshots = _prepare_run(tmp_path, checkpoints=3)
    blocked_source = snapshots[2] / "state.pkl"
    blocked_state = blocked_source.read_bytes()
    blocked_snapshot = (snapshots[2] / "run" / "snapshot.bin").read_bytes()
    request_bytes = (run / "wire" / "0001.request.json").read_bytes()
    blocked = run / "attempts" / "1" / "checkpoints" / snapshots[2].name / "state.pkl"
    original_unlink = Path.unlink
    error = PermissionError(errno.EACCES, "permission denied", str(blocked))
    already_failed = False

    def deny_once(path, *args, **kwargs):
        nonlocal already_failed
        if path == blocked and not already_failed:
            already_failed = True
            raise error
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_once)
    archive = _archive(run, spec)

    _assert_resume_completed(run, archive)
    archived_root = archive / "checkpoints"
    assert not (archived_root / snapshots[1].name / "state.pkl").exists()
    assert (archived_root / snapshots[2].name / "state.pkl").read_bytes() == blocked_state
    assert (archived_root / snapshots[2].name / "run" / "snapshot.bin").read_bytes() == blocked_snapshot
    assert not (archived_root / snapshots[3].name / "state.pkl").exists()
    assert (archive / "wire" / "0001.request.json").read_bytes() == request_bytes == _REQUEST
    saved_spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    assert saved_spec["resume"]["tape"][0]["request"] == str(archive / "wire" / "0001.request.json")
    _assert_one_diagnostic(capsys, run, "PermissionError", "EACCES")

    resume._compact_archived_checkpoints(archived_root)
    assert not (archived_root / snapshots[2].name / "state.pkl").exists()
    assert not (archived_root / snapshots[2].name / "run").exists()


def test_later_resume_retries_prior_archived_checkpoint_cleanup(tmp_path, monkeypatch, capsys):
    run, spec, snapshots = _prepare_run(tmp_path)
    blocked = run / "attempts" / "1" / "checkpoints" / snapshots[1].name / "state.pkl"
    original_unlink = Path.unlink
    error = PermissionError(errno.EACCES, "permission denied", str(blocked))
    failed_once = False

    def deny_once(path, *args, **kwargs):
        nonlocal failed_once
        if path == blocked and not failed_once:
            failed_once = True
            raise error
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_once)
    first_archive = _archive(run, spec)
    assert (first_archive / "checkpoints" / snapshots[1].name / "state.pkl").exists()

    wire = run / "wire"
    wire.mkdir()
    (wire / "0001.request.json").write_bytes(_REQUEST)
    (wire / "0001.response.json").write_bytes(_RESPONSE)
    (run / "sends.jsonl").write_text("", encoding="utf-8")
    second_spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    second_archive = resume.archive_attempt({
        "run": run,
        "spec": second_spec,
        "attempt": 2,
        "max_turns": 8,
        "previous_status": "interrupted",
    }, run.parent / "engine", use_checkpoint=False)

    assert second_archive == run / "attempts" / "2"
    assert not (first_archive / "checkpoints" / snapshots[1].name / "state.pkl").exists()
    captured = capsys.readouterr()
    assert captured.err.count("checkpoint cleanup PermissionError EACCES") == 1


def test_read_only_filesystem_during_payload_removal_does_not_stop_resume(
    tmp_path, monkeypatch, capsys
):
    run, spec, snapshots = _prepare_run(tmp_path)
    checkpoint = snapshots[1]
    source_payload = checkpoint / "run"
    payload = run / "attempts" / "1" / "checkpoints" / checkpoint.name / "run"
    original = (source_payload / "snapshot.bin").read_bytes()
    original_rmtree = shutil.rmtree
    error = OSError(errno.EROFS, "read-only filesystem", str(payload))

    def reject_payload(path, *args, **kwargs):
        if Path(path) == payload:
            raise error
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", reject_payload)
    archive = _archive(run, spec)

    _assert_resume_completed(run, archive)
    archived_checkpoint = archive / "checkpoints" / checkpoint.name
    assert not (archived_checkpoint / "state.pkl").exists()
    assert (archived_checkpoint / "run" / "snapshot.bin").read_bytes() == original
    assert (archived_checkpoint / "checkpoint.json").is_file()
    _assert_one_diagnostic(capsys, run, "OSError", "EROFS")


@pytest.mark.parametrize("target_name", ["blob-pool", "journal-cache", ".partial.tmp", "LATEST"])
def test_interruption_at_each_remaining_cleanup_step_does_not_stop_resume(
    tmp_path, monkeypatch, capsys, target_name
):
    run, spec, _snapshots = _prepare_run(tmp_path)
    target = run / "attempts" / "1" / "checkpoints" / target_name
    source_target = run / "checkpoints" / target_name
    if target_name == "LATEST":
        original_bytes = source_target.read_bytes()
        original_unlink = Path.unlink
        error = OSError(errno.EIO, "simulated interruption", str(target))

        def interrupt_unlink(path, *args, **kwargs):
            if path == target:
                raise error
            return original_unlink(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", interrupt_unlink)
    else:
        original_bytes = (source_target / "saved.bin").read_bytes()
        original_rmtree = shutil.rmtree
        error = OSError(errno.EIO, "simulated interruption", str(target))

        def interrupt_rmtree(path, *args, **kwargs):
            if Path(path) == target:
                raise error
            return original_rmtree(path, *args, **kwargs)

        monkeypatch.setattr(shutil, "rmtree", interrupt_rmtree)

    archive = _archive(run, spec)

    _assert_resume_completed(run, archive)
    archived_target = archive / "checkpoints" / target_name
    if target_name == "LATEST":
        assert archived_target.read_bytes() == original_bytes
    else:
        assert (archived_target / "saved.bin").read_bytes() == original_bytes
    _assert_one_diagnostic(capsys, run, "OSError", "EIO")


@pytest.mark.parametrize("mask, expected_mode", [(0o002, 0o775), (0o022, 0o755)])
def test_checkpoint_compaction_keeps_base_mode_behavior_under_umask(
    tmp_path, mask, expected_mode
):
    previous = os.umask(mask)
    try:
        run, spec, snapshots = _prepare_run(tmp_path)
        checkpoint_mode = stat.S_IMODE(snapshots[1].stat().st_mode)
        request_bytes = (run / "wire" / "0001.request.json").read_bytes()
        archive = _archive(run, spec)
    finally:
        os.umask(previous)

    _assert_resume_completed(run, archive)
    archived = archive / "checkpoints" / snapshots[1].name
    assert checkpoint_mode == expected_mode
    assert stat.S_IMODE(archived.stat().st_mode) == expected_mode
    assert (archived / "checkpoint.json").is_file()
    assert not (archived / "state.pkl").exists()
    assert not (archived / "run").exists()
    assert (archive / "wire" / "0001.request.json").read_bytes() == request_bytes == _REQUEST
