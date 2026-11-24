"""Regression coverage for retrying a partially published chat session."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from stateframe import chat, workcopy


_REPLACEMENTS = ("a_modify.py", "b_mode.py", "n_create.py")
_DELETED = "z_delete.py"
_BASELINE = {
    "a_modify.py": b"old a\n",
    "b_mode.py": b"old b\n",
    "keep.txt": b"keep\n",
    "z_delete.py": b"old z\n",
}
_CANDIDATE = {
    "a_modify.py": b"new a\n",
    "b_mode.py": b"new b\n",
    "keep.txt": b"keep\n",
    "n_create.py": b"new n\n",
}


def _session(tmp_path: Path, *, hooks_enabled: bool = False) -> chat.Session:
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, data in _BASELINE.items():
        (repo / rel).write_bytes(data)
    (repo / "b_mode.py").chmod(0o640)

    session = chat.Session(repo=repo, home=tmp_path / "session", base=None)
    session.start(printer=lambda *_: None)
    (session.copy / "a_modify.py").write_bytes(_CANDIDATE["a_modify.py"])
    (session.copy / "b_mode.py").write_bytes(_CANDIDATE["b_mode.py"])
    (session.copy / "n_create.py").write_bytes(_CANDIDATE["n_create.py"])
    (session.copy / _DELETED).unlink()
    if hooks_enabled:
        # Exercise Session.apply's hook-enabled change discovery without running hooks.
        session._project_hooks = object()
    return session


def _tree_state(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in root.rglob("*") if path.is_file()
    }


def _restart(session: chat.Session, *, hooks_enabled: bool = False) -> chat.Session:
    baseline = {rel: hashlib.sha256(data).hexdigest()
                for rel, (data, _mode) in _tree_state(session.start_snapshot).items()}
    restarted = chat.Session(repo=session.repo, home=session.home, base=None, baseline=baseline)
    restarted._active_copy = session.copy
    restarted._baseline_snapshot = session.start_snapshot
    if hooks_enabled:
        restarted._project_hooks = object()
    return restarted


def _fault(kind: str) -> None:
    if kind == "enospc":
        raise OSError(errno.ENOSPC, "injected storage failure")
    raise KeyboardInterrupt


def test_large_text_survives_chat_promotion_and_apply_without_loading_unchanged_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    original = "".join(f"row {index:07d}\n" for index in range(120_000))
    changed = original.replace("row 0000005\n", "row changed\n")
    (repo / "large.txt").write_text(original)
    (repo / "steady.txt").write_text("steady\n" * 200_000)
    session = chat.Session(repo=repo, home=tmp_path / "session", base=None)
    session.start(printer=lambda *_: None)

    run = session.home / "task-01"
    workspace = run / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "large.txt").write_text(changed)
    record = chat.TaskRecord("change large file", "DONE", True, ["large.txt"], "",
                             str(run), False, attempt=1)
    read_bytes = Path.read_bytes
    unchanged_paths = {repo / "steady.txt", session.copy / "steady.txt",
                       session.start_snapshot / "steady.txt"}

    def reject_unchanged(path):
        if path in unchanged_paths:
            raise AssertionError("unchanged large files must not be loaded")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unchanged)
    session._promote(record)
    assert "+row changed" in session.session_diff()
    assert session.apply() == ["large.txt"]
    assert (repo / "large.txt").read_text() == changed


@pytest.mark.parametrize("hooks_enabled", [False, True])
@pytest.mark.parametrize(
    "operation,index,phase,failure",
    [
        ("replace", index, phase, failure)
        for index in (1, 2, 3)
        for phase in ("before", "after")
        for failure in ("enospc", "interrupt")
    ] + [("unlink", 1, "after", failure) for failure in ("enospc", "interrupt")],
)
def test_retry_publishes_only_unfinished_paths_after_each_interruption(
    tmp_path, monkeypatch, hooks_enabled, operation, index, phase, failure,
):
    session = _session(tmp_path, hooks_enabled=hooks_enabled)
    original_replace, original_unlink = workcopy.os.replace, workcopy.os.unlink
    first_replacements: list[str] = []
    first_unlinks: list[str] = []
    replacement_index = 0

    def interrupted_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is None:
            return original_replace(source, target, *args, **kwargs)
        replacement_index += 1
        if operation == "replace" and replacement_index == index and phase == "before":
            _fault(failure)
        result = original_replace(source, target, *args, **kwargs)
        first_replacements.append(str(target))
        if operation == "replace" and replacement_index == index and phase == "after":
            _fault(failure)
        return result

    def interrupted_unlink(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and str(path) == _DELETED:
            if operation == "unlink":
                original_unlink(path, *args, **kwargs)
                first_unlinks.append(str(path))
                _fault(failure)
        return original_unlink(path, *args, **kwargs)

    error = KeyboardInterrupt if failure == "interrupt" else workcopy.WorkCopyError
    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", interrupted_replace)
        patch.setattr(workcopy.os, "unlink", interrupted_unlink)
        with pytest.raises(error):
            session.apply()

    published_replacements = index if operation == "replace" and phase == "after" else index - 1
    if operation == "unlink":
        published_replacements = len(_REPLACEMENTS)
        assert first_unlinks == [_DELETED]
    assert first_replacements == list(_REPLACEMENTS[:published_replacements])

    session = _restart(session, hooks_enabled=hooks_enabled)

    if operation == "replace" and phase == "before":
        pending = list(_REPLACEMENTS[index - 1:]) + [_DELETED]
    elif operation == "replace" and phase == "after":
        pending = list(_REPLACEMENTS[index:]) + [_DELETED]
    else:
        pending = []

    retry_replacements: list[str] = []
    retry_unlinks: list[str] = []

    def observe_replace(source, target, *args, **kwargs):
        if kwargs.get("dst_dir_fd") is not None:
            retry_replacements.append(str(target))
        return original_replace(source, target, *args, **kwargs)

    def observe_unlink(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and str(path) == _DELETED:
            retry_unlinks.append(str(path))
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", observe_replace)
        patch.setattr(workcopy.os, "unlink", observe_unlink)
        assert session.apply() == pending

    assert retry_replacements == [path for path in pending if path != _DELETED]
    assert retry_unlinks == ([_DELETED] if _DELETED in pending else [])
    final = _tree_state(session.repo)
    assert {rel: data for rel, (data, _mode) in final.items()} == _CANDIDATE
    assert final["b_mode.py"][1] == 0o640


@pytest.mark.parametrize("hooks_enabled", [False, True])
def test_retry_accepts_exact_published_state_but_refuses_an_unpublished_user_edit(
    tmp_path, monkeypatch, hooks_enabled,
):
    session = _session(tmp_path, hooks_enabled=hooks_enabled)
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    (session.repo / "b_mode.py").write_bytes(b"user edit\n")
    before_retry = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError) as caught:
        session.apply()
    assert "b_mode.py" in str(caught.value)
    assert "a_modify.py" not in str(caught.value)
    assert _tree_state(session.repo) == before_retry


def test_user_edit_to_published_file_conflicts_and_does_not_clear_the_attempt(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    edited = b"user's published-file edit\n"
    (session.repo / "a_modify.py").write_bytes(edited)
    before_retry = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError) as caught:
        session.apply()
    assert "a_modify.py" in str(caught.value)
    assert _tree_state(session.repo) == before_retry

    (session.repo / "a_modify.py").write_bytes(_CANDIDATE["a_modify.py"])
    session.apply()
    assert {rel: data for rel, (data, _mode) in _tree_state(session.repo).items()} == _CANDIDATE


def test_user_mode_change_to_published_file_is_a_conflict(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace
    replacement_index = 0

    def fail_after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", fail_after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    destination = session.repo / "a_modify.py"
    destination.chmod(stat.S_IMODE(destination.stat().st_mode) ^ stat.S_IXUSR)
    before_retry = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError) as caught:
        session.apply()
    assert "a_modify.py" in str(caught.value)
    assert _tree_state(session.repo) == before_retry
    with pytest.raises(workcopy.WorkCopyError) as caught:
        session.apply()
    assert "a_modify.py" in str(caught.value)
    assert _tree_state(session.repo) == before_retry


def test_mode_only_change_on_unpublished_baseline_keeps_existing_behavior(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace

    def fail_before_replace(source, target, *args, **kwargs):
        if kwargs.get("dst_dir_fd") is not None:
            raise OSError(errno.ENOSPC, "injected storage failure")
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", fail_before_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    destination = session.repo / "a_modify.py"
    destination.chmod(0o600)
    session.apply()
    assert destination.read_bytes() == _CANDIDATE["a_modify.py"]
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_withdrawn_candidate_path_is_restored_to_the_session_baseline(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    (session.copy / "a_modify.py").write_bytes(_BASELINE["a_modify.py"])
    session.apply()
    assert (session.repo / "a_modify.py").read_bytes() == _BASELINE["a_modify.py"]


def test_retry_uses_new_candidate_content_for_an_unpublished_path(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    newer = b"newer candidate\n"
    (session.copy / "b_mode.py").write_bytes(newer)
    session.apply()
    assert (session.repo / "a_modify.py").read_bytes() == _CANDIDATE["a_modify.py"]
    assert (session.repo / "b_mode.py").read_bytes() == newer


def test_attempt_write_failure_publishes_nothing(tmp_path, monkeypatch):
    session = _session(tmp_path)
    before = _tree_state(session.repo)
    original_replace = workcopy.os.replace

    def fail_session_state_write(source, target, *args, **kwargs):
        if kwargs.get("dst_dir_fd") is None:
            raise OSError(errno.ENOSPC, "injected attempt write failure")
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", fail_session_state_write)
        with pytest.raises((OSError, workcopy.WorkCopyError)):
            session.apply()
    assert _tree_state(session.repo) == before


def test_apply_attempt_survives_a_session_process_restart(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    attempt_path = session.home / "apply-attempt.json"
    attempt_bytes = attempt_path.read_bytes()
    attempt = json.loads(attempt_bytes)
    assert stat.S_IMODE(attempt_path.stat().st_mode) == 0o600
    assert b"new a" not in attempt_bytes
    assert len(attempt["paths"]["a_modify.py"]) == 1
    assert set(attempt["paths"]["a_modify.py"][0]) == {"sha256", "mode"}

    restarted = _restart(session)
    assert restarted.apply() == ["b_mode.py", "n_create.py", _DELETED]
    assert {rel: data for rel, (data, _mode) in _tree_state(session.repo).items()} == _CANDIDATE


def test_missing_attempt_file_uses_the_session_memory_copy(tmp_path, monkeypatch):
    session = _session(tmp_path)
    original_replace = workcopy.os.replace

    def interrupt_after_first(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if kwargs.get("dst_dir_fd") is not None and str(target) == _REPLACEMENTS[0]:
            raise OSError(errno.ENOSPC, "injected storage failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", interrupt_after_first)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    (session.home / "apply-attempt.json").unlink()
    assert session.apply() == [*_REPLACEMENTS[1:], _DELETED]
    assert {rel: data for rel, (data, _mode) in _tree_state(session.repo).items()} == _CANDIDATE


def test_retry_accepts_new_file_at_the_mode_guarded_apply_creates(tmp_path, monkeypatch):
    old_umask = os.umask(0o022)
    try:
        session = _session(tmp_path)
        (session.copy / "n_create.py").chmod(0o600)
        original_replace = workcopy.os.replace

        def interrupt_after_create(source, target, *args, **kwargs):
            result = original_replace(source, target, *args, **kwargs)
            if kwargs.get("dst_dir_fd") is not None and str(target) == "n_create.py":
                raise KeyboardInterrupt
            return result

        with monkeypatch.context() as patch:
            patch.setattr(workcopy.os, "replace", interrupt_after_create)
            with pytest.raises(KeyboardInterrupt):
                session.apply()

        assert stat.S_IMODE((session.repo / "n_create.py").stat().st_mode) == 0o644
        assert session.apply() == [_DELETED]
        assert (session.repo / "n_create.py").read_bytes() == _CANDIDATE["n_create.py"]
    finally:
        os.umask(old_umask)


def test_persisted_attempt_paths_are_validated_before_snapshot_reads(tmp_path, monkeypatch):
    session = _session(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"private test value")
    attempt_path = session.home / "apply-attempt.json"
    paths = {"../../outside": {"sha256": "0" * 64, "mode": 0o600}}
    attempt_path.write_text(json.dumps({
        "schema": chat._APPLY_ATTEMPT_SCHEMA, "paths": paths,
        "checksum": chat._attempt_checksum(paths),
    }), encoding="utf-8")
    original_read_bytes = Path.read_bytes
    notices = []

    def reject_outside_snapshot(path):
        if path == outside:
            raise AssertionError("attempt path read outside the session snapshot")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_outside_snapshot)
    assert session.apply(printer=notices.append) == [*_REPLACEMENTS, _DELETED]
    assert notices == ["notice: ignored invalid apply-attempt.json"]
    assert original_read_bytes(outside) == b"private test value"


def test_recreating_a_deleted_path_with_baseline_bytes_is_deleted_again(tmp_path, monkeypatch):
    session = _session(tmp_path)
    (session.copy / "a_modify.py").write_bytes(_BASELINE["a_modify.py"])
    (session.copy / "b_mode.py").write_bytes(_BASELINE["b_mode.py"])
    (session.copy / "n_create.py").unlink()
    original_unlink = workcopy.os.unlink

    def after_delete(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and str(path) == _DELETED:
            original_unlink(path, *args, **kwargs)
            raise KeyboardInterrupt
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "unlink", after_delete)
        with pytest.raises(KeyboardInterrupt):
            session.apply()

    (session.repo / _DELETED).write_bytes(_BASELINE[_DELETED])
    assert session.apply() == [_DELETED]
    assert not (session.repo / _DELETED).exists()


def test_recreating_a_deleted_path_with_different_bytes_conflicts(tmp_path, monkeypatch):
    session = _session(tmp_path)
    (session.copy / "a_modify.py").write_bytes(_BASELINE["a_modify.py"])
    (session.copy / "b_mode.py").write_bytes(_BASELINE["b_mode.py"])
    (session.copy / "n_create.py").unlink()
    original_unlink = workcopy.os.unlink

    def after_delete(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and str(path) == _DELETED:
            original_unlink(path, *args, **kwargs)
            raise KeyboardInterrupt
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "unlink", after_delete)
        with pytest.raises(KeyboardInterrupt):
            session.apply()

    (session.repo / _DELETED).write_bytes(b"user recreation\n")
    before_retry = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError) as caught:
        session.apply()
    assert _DELETED in str(caught.value)
    assert _tree_state(session.repo) == before_retry


@pytest.mark.parametrize("hooks_enabled", [False, True])
def test_interrupted_apply_does_not_change_model_visible_session_facts(
    tmp_path, monkeypatch, hooks_enabled,
):
    session = _session(tmp_path, hooks_enabled=hooks_enabled)
    session.history.append(chat.TaskRecord(
        task="previous task", result="SUBMITTED", tests_passed=True, files=["a_modify.py"],
        diff="", run_dir="", kept=True,
    ))
    request_text_before = session.task_text("next task")
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    assert session.task_text("next task") == request_text_before


def test_in_place_apply_keeps_its_existing_nonretrying_behavior(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_bytes(b"old a\n")
    (repo / "b.py").write_bytes(b"old b\n")
    work = workcopy.create(repo, tmp_path / "work")
    (work.copy / "a.py").write_bytes(b"new a\n")
    (work.copy / "b.py").write_bytes(b"new b\n")
    original_replace = workcopy.os.replace
    replacement_index = 0

    def after_first_replace(source, target, *args, **kwargs):
        nonlocal replacement_index
        if kwargs.get("dst_dir_fd") is not None:
            replacement_index += 1
            result = original_replace(source, target, *args, **kwargs)
            if replacement_index == 1:
                raise OSError(errno.ENOSPC, "injected storage failure")
            return result
        return original_replace(source, target, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", after_first_replace)
        with pytest.raises(workcopy.WorkCopyError):
            workcopy.apply_in_place(work)

    with pytest.raises(workcopy.WorkCopyError):
        workcopy.apply_in_place(work)
    assert (repo / "a.py").read_bytes() == b"new a\n"
    assert (repo / "b.py").read_bytes() == b"old b\n"


def test_apply_refuses_symlink_with_existing_text_without_following_its_target(tmp_path, monkeypatch):
    session = _session(tmp_path)
    destination = session.repo / "a_modify.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"private external bytes\n")
    destination.unlink()
    destination.symlink_to(outside)
    original_stat = os.stat
    followed_links = []

    def reject_following(path, *args, **kwargs):
        if not isinstance(path, int) and Path(path) == destination and kwargs.get("dir_fd") is None:
            followed_links.append(path)
            raise AssertionError("destination lookup followed a link outside the repository")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "stat", reject_following)
        with pytest.raises(workcopy.WorkCopyError) as caught:
            session.apply()

    assert followed_links == []
    assert str(caught.value) == (
        "refusing /apply: a_modify.py (destination is a symlink: a_modify.py)"
    )
    assert outside.read_bytes() == b"private external bytes\n"


@pytest.mark.parametrize("corruption", ["unreadable", "non_json", "wrong_schema", "bad_value", "unknown_path", "tampered"])
def test_invalid_apply_attempt_is_ignored_with_one_notice_and_baseline_conflict(
    tmp_path, monkeypatch, corruption,
):
    session = _session(tmp_path)
    conflict = session.repo / "a_modify.py"
    conflict.write_bytes(b"user bytes\n")
    before = _tree_state(session.repo)
    attempt_path = session.home / "apply-attempt.json"
    def record(paths):
        return {"schema": chat._APPLY_ATTEMPT_SCHEMA, "paths": paths,
                "checksum": chat._attempt_checksum(paths)}

    valid = record({})
    if corruption == "non_json":
        attempt_path.write_text("{not-json", encoding="utf-8")
    elif corruption == "wrong_schema":
        attempt_path.write_text(json.dumps({**valid, "schema": "wrong"}), encoding="utf-8")
    elif corruption == "bad_value":
        attempt_path.write_text(json.dumps(record({"a_modify.py": {"sha256": "not-a-hash", "mode": -1}})),
                                encoding="utf-8")
    elif corruption == "unknown_path":
        attempt_path.write_text(json.dumps(record({"../../outside.py": None})), encoding="utf-8")
    elif corruption == "tampered":
        paths = {"a_modify.py": {"sha256": hashlib.sha256(_CANDIDATE["a_modify.py"]).hexdigest(),
                                 "mode": stat.S_IMODE(conflict.stat().st_mode)}}
        tampered = record(paths)
        paths["a_modify.py"]["sha256"] = hashlib.sha256(b"user bytes\n").hexdigest()
        attempt_path.write_text(json.dumps(tampered), encoding="utf-8")
    else:
        attempt_path.write_text(json.dumps(valid), encoding="utf-8")
        original_open = os.open

        def unreadable(path, flags, *args, **kwargs):
            if path == attempt_path:
                raise PermissionError("injected unreadable attempt")
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(chat.os, "open", unreadable)

    notices = []
    with pytest.raises(workcopy.WorkCopyError) as caught:
        session.apply(printer=notices.append)

    assert str(caught.value) == "refusing /apply: a_modify.py (changed in the original since the start)"
    assert notices == ["notice: ignored invalid apply-attempt.json"]
    assert _tree_state(session.repo) == before


def test_new_file_mode_tracking_does_not_toggle_the_process_umask(tmp_path, monkeypatch):
    session = _session(tmp_path)
    (session.copy / "n_create.py").chmod(0o600)
    original_replace = workcopy.os.replace

    def interrupt_after_create(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if kwargs.get("dst_dir_fd") is not None and str(target) == "n_create.py":
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch.setattr(chat.os, "umask", lambda *_: pytest.fail("apply toggled the process umask"))
        patch.setattr(workcopy.os, "replace", interrupt_after_create)
        with pytest.raises(KeyboardInterrupt):
            session.apply()

    attempt = json.loads((session.home / "apply-attempt.json").read_text(encoding="utf-8"))
    assert set(attempt["paths"]["n_create.py"][0]) == {"sha256"}
    assert session.apply() == [_DELETED]


def test_apply_attempts_are_scoped_to_their_own_sessions(tmp_path, monkeypatch):
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = _session(first_root)
    second = chat.Session(repo=first.repo, home=tmp_path / "second-session", base=None)
    second.start(printer=lambda *_: None)
    (second.copy / "a_modify.py").write_bytes(_CANDIDATE["a_modify.py"])
    (second.copy / "b_mode.py").write_bytes(_CANDIDATE["b_mode.py"])
    (second.copy / "n_create.py").write_bytes(_CANDIDATE["n_create.py"])
    (second.copy / _DELETED).unlink()
    original_replace = workcopy.os.replace

    def interrupt_after_first(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if kwargs.get("dst_dir_fd") is not None and str(target) == "a_modify.py":
            raise OSError(errno.ENOSPC, "injected storage failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", interrupt_after_first)
        with pytest.raises(workcopy.WorkCopyError):
            first.apply()

    before_second = _tree_state(first.repo)
    with pytest.raises(workcopy.WorkCopyError) as caught:
        second.apply()
    assert str(caught.value) == "refusing /apply: a_modify.py (changed in the original since the start)"
    assert _tree_state(first.repo) == before_second


def _single_modify_session(tmp_path):
    session = _session(tmp_path)
    (session.copy / "b_mode.py").write_bytes(_BASELINE["b_mode.py"])
    (session.copy / "n_create.py").unlink()
    (session.copy / _DELETED).write_bytes(_BASELINE[_DELETED])
    return session


def _interrupt_modify(session, monkeypatch):
    original_replace = workcopy.os.replace

    def interrupt(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if kwargs.get("dst_dir_fd") is not None and str(target) == "a_modify.py":
            raise OSError(errno.ENOSPC, "injected storage failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", interrupt)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()


def _user_tree_state(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
        for path in root.rglob("*")
        if path.is_file() and path.suffix != ".tmp"
    }


def _replace_parent(root: Path, home: Path, name: str, replacement_kind: str,
                    user_bytes: bytes) -> tuple[Path, Path | None]:
    moved = home / "moved-parent"
    root.rename(moved)
    if replacement_kind == "moved":
        replacement = None
    elif replacement_kind == "symlink":
        replacement = home / "replacement-parent"
        replacement.mkdir()
        (replacement / name).write_bytes(user_bytes)
        root.symlink_to(replacement, target_is_directory=True)
    else:
        replacement = root
        replacement.mkdir()
        (replacement / name).write_bytes(user_bytes)
    return moved, replacement


@pytest.mark.parametrize("operation", ["replace", "delete"])
@pytest.mark.parametrize("retry_state", [False, True], ids=["without-retry", "with-retry"])
@pytest.mark.parametrize("replacement_kind", ["moved", "symlink", "directory"])
def test_parent_identity_change_refuses_before_mutation(
    tmp_path, monkeypatch, operation, retry_state, replacement_kind,
):
    rel = "a_modify.py"
    if retry_state:
        session = _single_modify_session(tmp_path)
        _interrupt_modify(session, monkeypatch)
        original_bytes = _CANDIDATE[rel]
        if operation == "replace":
            (session.copy / rel).write_bytes(b"newer candidate\n")
        else:
            (session.copy / rel).unlink()
        root, home = session.repo, session.home
        assert (home / "apply-attempt.json").is_file()
    else:
        root, home = tmp_path / "repo", tmp_path
        root.mkdir()
        original_bytes = b"baseline bytes\n"
        (root / rel).write_bytes(original_bytes)
        changes = {rel: b"candidate bytes\n" if operation == "replace" else None}
        baseline = {rel: hashlib.sha256(original_bytes).hexdigest()}

    swap: dict[str, object] = {}
    original_read_at = workcopy._read_at
    reads_before_mutation = 3 if operation == "replace" else 2
    read_count = 0

    def swap_after_final_leaf_read(parent, name):
        nonlocal read_count
        result = original_read_at(parent, name)
        if name == rel:
            read_count += 1
            if read_count == reads_before_mutation:
                moved, replacement = _replace_parent(
                    root, home, rel, replacement_kind, b"replacement user bytes\n")
                swap["moved"] = moved
                swap["replacement"] = replacement
                swap["moved_state"] = _user_tree_state(moved)
                if replacement is not None:
                    swap["replacement_state"] = _user_tree_state(replacement)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(workcopy, "_read_at", swap_after_final_leaf_read)
        with pytest.raises(workcopy.WorkCopyError) as caught:
            if retry_state:
                session.apply()
            else:
                workcopy.guarded_apply(root, changes, baseline, what="test apply")

    assert "already applied: nothing" in str(caught.value)
    assert str(tmp_path) not in str(caught.value)
    assert read_count == reads_before_mutation
    moved = swap["moved"]
    assert isinstance(moved, Path)
    assert _user_tree_state(moved) == swap["moved_state"]
    assert (moved / rel).read_bytes() == original_bytes
    replacement = swap["replacement"]
    if replacement_kind == "moved":
        assert replacement is None
        assert not root.exists()
    else:
        assert isinstance(replacement, Path)
        assert _user_tree_state(replacement) == swap["replacement_state"]
        assert (replacement / rel).read_bytes() == b"replacement user bytes\n"
        if replacement_kind == "symlink":
            assert root.is_symlink()
        else:
            assert root.is_dir() and not root.is_symlink()
    assert not [path for path in moved.iterdir() if path.suffix == ".tmp"]
    if retry_state:
        assert (home / "apply-attempt.json").is_file()


def test_new_file_retry_ignores_umask_changes_and_preserves_current_mode(tmp_path, monkeypatch):
    previous_umask = os.umask(0o022)
    try:
        session = _session(tmp_path)
        original_replace = workcopy.os.replace

        def interrupt_after_create(source, target, *args, **kwargs):
            result = original_replace(source, target, *args, **kwargs)
            if kwargs.get("dst_dir_fd") is not None and str(target) == "n_create.py":
                raise KeyboardInterrupt
            return result

        with monkeypatch.context() as patch:
            patch.setattr(workcopy.os, "replace", interrupt_after_create)
            with pytest.raises(KeyboardInterrupt):
                session.apply()

        attempt = json.loads((session.home / "apply-attempt.json").read_text(encoding="utf-8"))
        assert set(attempt["paths"]["n_create.py"][0]) == {"sha256"}
        assert not (session.home / ".apply-mode").exists()
        created = session.repo / "n_create.py"
        created.write_bytes(b"user edit after interrupted create\n")
        before_edit = _tree_state(session.repo)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()
        assert _tree_state(session.repo) == before_edit

        created.write_bytes(_CANDIDATE["n_create.py"])
        created.chmod(0o604)
        (session.copy / "n_create.py").write_bytes(b"newest candidate\n")
        current_umask = os.umask(0o077)
        try:
            assert session.apply() == ["n_create.py", _DELETED]
        finally:
            os.umask(current_umask)
        assert created.read_bytes() == b"newest candidate\n"
        assert stat.S_IMODE(created.stat().st_mode) == 0o604
    finally:
        os.umask(previous_umask)


@pytest.mark.parametrize("artifact", ["directory", "file", "symlink"])
def test_apply_mode_path_has_no_effect_on_retry(tmp_path, artifact):
    session = _single_modify_session(tmp_path)
    path = session.home / ".apply-mode"
    outside = tmp_path / "outside"
    outside.write_bytes(b"private sentinel\n")
    if artifact == "directory":
        path.mkdir()
    elif artifact == "file":
        path.write_bytes(b"sentinel\n")
    else:
        path.symlink_to(outside)

    assert session.apply() == ["a_modify.py"]
    assert (session.repo / "a_modify.py").read_bytes() == _CANDIDATE["a_modify.py"]
    if artifact == "directory":
        assert path.is_dir()
    elif artifact == "file":
        assert path.read_bytes() == b"sentinel\n"
    else:
        assert path.is_symlink()
        assert outside.read_bytes() == b"private sentinel\n"
    (session.repo / "a_modify.py").write_bytes(b"user edit after mode-path check\n")
    (session.copy / "a_modify.py").write_bytes(b"follow-up candidate\n")
    before_edit = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError):
        session.apply()
    assert _tree_state(session.repo) == before_edit


def test_two_session_handles_share_retry_state_safely(tmp_path, monkeypatch):
    first = _session(tmp_path)
    second = _restart(first)
    assert first.home == second.home
    original_replace = workcopy.os.replace

    def interrupt_first(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if kwargs.get("dst_dir_fd") is not None and str(target) == "a_modify.py":
            raise OSError(errno.ENOSPC, "injected storage failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", interrupt_first)
        with pytest.raises(workcopy.WorkCopyError):
            first.apply()

    assert second.apply() == ["b_mode.py", "n_create.py", _DELETED]
    edited = b"user edit after shared-home retry\n"
    (first.repo / "b_mode.py").write_bytes(edited)
    (second.copy / "b_mode.py").write_bytes(b"follow-up candidate\n")
    before = _tree_state(first.repo)
    with pytest.raises(workcopy.WorkCopyError):
        second.apply()
    assert _tree_state(first.repo) == before


def test_repeated_candidate_changes_keep_every_published_hash_until_success(tmp_path, monkeypatch):
    session = _single_modify_session(tmp_path)
    _interrupt_modify(session, monkeypatch)
    first = _CANDIDATE["a_modify.py"]
    second = b"second candidate\n"
    newest = b"newest candidate\n"
    (session.copy / "a_modify.py").write_bytes(second)
    original_replace = workcopy.os.replace

    def interrupt_second(source, target, *args, **kwargs):
        result = original_replace(source, target, *args, **kwargs)
        if kwargs.get("dst_dir_fd") is not None and str(target) == "a_modify.py":
            raise OSError(errno.ENOSPC, "injected storage failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", interrupt_second)
        with pytest.raises(workcopy.WorkCopyError):
            session.apply()

    path = session.home / "apply-attempt.json"
    attempt = json.loads(path.read_text(encoding="utf-8"))
    hashes = {item["sha256"] for item in attempt["paths"]["a_modify.py"] if item is not None}
    assert hashes == {hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()}

    edited = b"user edit after repeated interruption\n"
    destination = session.repo / "a_modify.py"
    destination.write_bytes(edited)
    before = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError):
        session.apply()
    assert _tree_state(session.repo) == before

    destination.write_bytes(second)
    (session.copy / "a_modify.py").write_bytes(newest)
    assert session.apply() == ["a_modify.py"]
    assert destination.read_bytes() == newest
    assert not path.exists()


def test_failed_attempt_unlink_clears_superseded_hashes(tmp_path, monkeypatch):
    session = _single_modify_session(tmp_path)
    _interrupt_modify(session, monkeypatch)
    (session.copy / "a_modify.py").write_bytes(b"completed candidate\n")
    attempt_path = session.home / "apply-attempt.json"
    original_unlink = Path.unlink

    def fail_attempt_unlink(path, *args, **kwargs):
        if path == attempt_path:
            raise OSError(errno.EACCES, "injected attempt cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_attempt_unlink)
        assert session.apply() == ["a_modify.py"]

    assert json.loads(attempt_path.read_text(encoding="utf-8"))["paths"] == {}
    destination = session.repo / "a_modify.py"
    destination.write_bytes(_CANDIDATE["a_modify.py"])
    (session.copy / "a_modify.py").write_bytes(b"next candidate\n")
    before = _tree_state(session.repo)
    with pytest.raises(workcopy.WorkCopyError):
        session.apply()
    assert _tree_state(session.repo) == before
