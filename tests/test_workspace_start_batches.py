"""Shell batches bind to the final revision; a full disk leaves no partial start copy."""
import errno
from functools import partial
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_shell_effect_uses_revision_after_file_changes_between_callbacks(tmp_path, monkeypatch):
    from kernel_green_lock import GreenLock
    from kernel_pilot import ACTIVE_RUNTIME
    from kernel_post_green_menu import note_test_receipt, pending_verification
    import kernel_read_reuse_visibility
    from kernel_shell_tool import publish_host_effects
    from kernel_stateframe_runtime import StateFrameRuntime

    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("FIRST = 1\n")
    second.write_text("SECOND = 1\n")

    def current_view(**_):
        revision = hashlib.sha256(first.read_bytes() + second.read_bytes()).hexdigest()
        return SimpleNamespace(as_binding=lambda: {
            "product_revision": revision, "environment_revision": "review",
        })

    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path), target=None, current_view=current_view,
        projector_kind="rich", journal=SimpleNamespace(manifest={}),
        _binding_cache=None, _binding_epoch=0, _unit_policy=None,
        _host_step_budget2_cache_enabled=StateFrameRuntime._host_step_budget2_cache_enabled,
        current_request_id="request_1", stateframe_post_green_work_menu_mode="on",
        stateframe_greenlock_menu_authority_mode="on",
    )
    runtime.binding = partial(StateFrameRuntime.binding, runtime)
    runtime.note_mutation = partial(StateFrameRuntime.note_mutation, runtime)
    runtime._invalidate_binding_cache = partial(StateFrameRuntime._invalidate_binding_cache, runtime)
    monkeypatch.setenv("STATEFRAME_GREEN_LOCK", "terminal")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "on")
    GreenLock(runtime, mode="terminal")
    bolt = SimpleNamespace(root=tmp_path, product_edits=0)
    previous = kernel_read_reuse_visibility.clear_candidates

    def between_effects(capture, *, path=None):
        if Path(path).name == second.name:
            first.write_text("FIRST = 2\n")
            StateFrameRuntime._note_applied_mutation(runtime, (first.name,))
        return previous(capture, path=path)

    monkeypatch.setattr(kernel_read_reuse_visibility, "clear_candidates", between_effects)
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        publish_host_effects(runtime, bolt, set(), changed=(first.name, second.name),
                             created_paths=(), name="")
    finally:
        ACTIVE_RUNTIME.reset(token)

    assert pending_verification(runtime)
    tested_revision = runtime.binding()
    pending = runtime._post_green_menu_state["pending"]["binding"]
    note_test_receipt(runtime, {
        "request_id": "request_2", "exit_code": 0, "binding_stable": True,
        "binding": tested_revision, "after_binding": tested_revision,
    })
    assert not pending_verification(runtime), (pending, tested_revision["artifact"])


@pytest.mark.parametrize("snapshot", [False])
def test_startup_cleans_copies_when_parent_copy_hits_enospc(tmp_path, monkeypatch, snapshot):
    from stateframe import runner
    from stateframe import workspace_start_snapshot
    from stateframe.endpoint import Endpoint

    source = tmp_path / "repo"
    source.mkdir()
    (source / "answer.py").write_text("VALUE = 1\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.delenv("STATEFRAME_SEND1_WAVE4", raising=False)
    run = tmp_path / "run"
    options = runner.RunOptions(
        source, "Change answer.py", Endpoint("http://localhost:65534/v1", "scripted"),
        run_dir=run, no_powerups=True, no_project_hooks=True,
    )
    options._send1_wave4 = snapshot
    calls = []

    def fill_disk(self, src, dst, info, *, data=None, writable=False, before_write=None):
        if Path(dst) == run / "parent" / "answer.py":
            calls.append(Path(dst))
            with Path(dst).open("xb") as stream:
                stream.write(b"V")
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return original_file(self, src, dst, info, data=data, writable=writable,
                             before_write=before_write)

    original_file = workspace_start_snapshot._Copies.file
    monkeypatch.setattr(workspace_start_snapshot._Copies, "file", fill_disk)
    with pytest.raises((OSError, runner.workcopy.WorkCopyError)) as raised:
        runner.prepare(options)
    assert calls == [run / "parent" / "answer.py"]
    leftovers = sorted(path.name for path in (run / "workspace", run / "parent") if path.exists())
    assert leftovers == [], (leftovers, str(raised.value))


def test_product_path_through_a_linked_directory_is_resolved(tmp_path):
    from kernel_green_lock import GreenLock

    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to("real", target_is_directory=True)
    lock = GreenLock.__new__(GreenLock)
    lock.runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    for path in ("link/x.py", "./link/x.py", tmp_path / "link" / "x.py", "real/x.py"):
        assert lock._workspace_path(path) == "real/x.py"


def _disk(monkeypatch, free, total=100 << 30):
    from stateframe import workcopy

    values = list(free) if isinstance(free, list) else [free]

    def statvfs(_path):
        value = values.pop(0) if len(values) > 1 else values[0]
        return SimpleNamespace(f_bavail=value, f_frsize=1, f_blocks=total)

    monkeypatch.setattr(workcopy.os, "statvfs", statvfs, raising=False)


def _repository(tmp_path, count, size=1 << 20):
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(count):
        with (repo / f"f{index:03}.txt").open("wb") as output:
            output.truncate(size)
    return repo


@pytest.mark.parametrize("snapshot", [False, True])
def test_large_repository_that_would_fill_the_disk_is_refused_before_copying(tmp_path, monkeypatch, snapshot):
    from stateframe import workcopy

    repo = _repository(tmp_path, 70)
    _disk(monkeypatch, 5 << 30)
    monkeypatch.setattr(workcopy, "_reflink",
                        lambda *_: (_ for _ in ()).throw(OSError(errno.EIO, "clone unavailable")))
    with pytest.raises(workcopy.WorkCopyError, match="to stay free") as error:
        workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=snapshot)
    assert "--repo" in str(error.value)
    assert not (tmp_path / "run").exists() or not list((tmp_path / "run").iterdir())


@pytest.mark.parametrize("snapshot", [False, True])
def test_small_repository_still_starts_on_a_nearly_full_disk(tmp_path, monkeypatch, snapshot):
    from stateframe import workcopy

    repo = _repository(tmp_path, 2, size=100)
    _disk(monkeypatch, 80 << 20)
    work = workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=snapshot)
    assert len(work.baseline) == 2


@pytest.mark.parametrize("snapshot", [False, True])
def test_copy_stops_and_cleans_up_when_the_disk_fills_meanwhile(tmp_path, monkeypatch, snapshot):
    from stateframe import workcopy

    repo = _repository(tmp_path, 150)
    _disk(monkeypatch, [50 << 30, 1 << 20])
    monkeypatch.setattr(workcopy, "_reflink",
                        lambda *_: (_ for _ in ()).throw(OSError(errno.EIO, "clone unavailable")))
    with pytest.raises(workcopy.WorkCopyError, match="needs"):
        workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=snapshot)
    assert not list((tmp_path / "run").iterdir())
    assert len(list(repo.iterdir())) == 150


def test_failed_parent_publication_keeps_a_manifest_it_did_not_write(tmp_path, monkeypatch):
    from stateframe import workcopy, workspace_start_snapshot

    repo = _repository(tmp_path, 1, size=10)
    work = workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=True)
    manifest = tmp_path / "run" / workspace_start_snapshot._MANIFEST
    manifest.write_text("not ours")

    def full(*_args):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(workspace_start_snapshot, "_private_file", full)
    with pytest.raises((OSError, workcopy.WorkCopyError)):
        workspace_start_snapshot.pin_parent(work, tmp_path / "run" / "parent")
    assert manifest.read_text() == "not ours"


@pytest.mark.parametrize("snapshot", [False, True])
def test_space_is_checked_where_the_run_directory_will_be_made(tmp_path, monkeypatch, snapshot):
    from stateframe import workcopy

    repo = _repository(tmp_path, 70)
    monkeypatch.setattr(workcopy, "_reflink",
                        lambda *_: (_ for _ in ()).throw(OSError(errno.EIO, "clone unavailable")))

    def statvfs(path):
        if not os.path.exists(path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
        assert Path(path) == tmp_path
        return SimpleNamespace(f_bavail=5 << 30, f_frsize=1, f_blocks=100 << 30)

    monkeypatch.setattr(workcopy.os, "statvfs", statvfs, raising=False)
    with pytest.raises(workcopy.WorkCopyError, match="to stay free"):
        workcopy.create(repo, tmp_path / "a" / "b" / "run" / "workspace", admit_snapshot=snapshot)
    assert not (tmp_path / "a").exists()


def test_space_rule_counts_both_copies_when_clone_probe_fails(tmp_path, monkeypatch):
    from stateframe import workcopy

    repo = _repository(tmp_path, 1)
    _disk(monkeypatch, (3 << 20) // 2, total=100 << 20)

    def refuse_clone(_source, _target):
        raise OSError(errno.EIO, "clone unavailable")

    monkeypatch.setattr(workcopy, "_reflink", refuse_clone)
    with pytest.raises(workcopy.WorkCopyError, match="needs 2097152 bytes"):
        workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=False)


def test_sparse_two_gib_file_is_refused_only_by_space_rule(tmp_path, monkeypatch):
    from stateframe import workcopy

    repo = _repository(tmp_path, 1, size=2 << 30)
    _disk(monkeypatch, 3 << 30, total=20 << 30)

    def refuse_clone(_source, _target):
        raise OSError(errno.EIO, "clone unavailable")

    monkeypatch.setattr(workcopy, "_reflink", refuse_clone)
    with pytest.raises(workcopy.WorkCopyError, match="needs 4294967296 bytes"):
        workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=False)


def test_resume_copy_checks_space_and_streams_large_baseline(tmp_path, monkeypatch):
    from stateframe import resume, runner, workcopy

    repo = tmp_path / "repo"
    repo.mkdir()
    run = tmp_path / "run"
    parent = run / "parent"
    parent.mkdir(parents=True)
    payload = parent / "payload.bin"
    payload.write_bytes(b"x" * (3 << 20))
    text = "".join(f"row {index:08d}\n" for index in range(90_000))
    assert len(text) > 1 << 20
    (parent / "large.txt").write_text(text)
    (parent / "payload-link").symlink_to(payload.name)
    checked = []
    check_space = workcopy._check_copy_space

    def record_check(*args, **kwargs):
        checked.append(args[0])
        return check_space(*args, **kwargs)

    monkeypatch.setattr(workcopy, "_check_copy_space", record_check)
    assert runner._fresh_resume_workcopy is resume.fresh_workcopy
    resumed = runner._fresh_resume_workcopy(run, repo, preserve_symlinks=True)

    assert checked == [parent]
    assert resumed.large_files == {"payload.bin", "large.txt"}
    assert resumed.baseline["payload.bin"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert (resumed.copy / "payload-link").is_symlink()
    (resumed.copy / "large.txt").write_text(text.replace("row 00000005\n", "row changed\n"))
    assert "+row changed" in workcopy.unified_diff(resumed)


@pytest.mark.parametrize("snapshot", [False, True])
def test_repository_under_the_large_size_never_needs_the_reserve(tmp_path, monkeypatch, snapshot):
    from stateframe import workcopy

    repo = _repository(tmp_path, 40)
    _disk(monkeypatch, 768 << 20)
    assert len(workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=snapshot).baseline) == 40


def test_skipped_summary_and_line_are_bounded(tmp_path):
    from stateframe import runner, workcopy

    work = workcopy.WorkCopy(original=tmp_path / "repo", copy=tmp_path / "copy")
    for index in range(205):
        work.note_skipped(f"path-{index:03}.txt", "special file")

    rows = runner._start_files_skipped_rows(work)
    line = runner._start_files_skipped_line(work)

    assert len(rows) == 200
    assert rows[0] == {"path": "path-000.txt", "reason": "special file"}
    assert "path-199.txt" in {row["path"] for row in rows}
    assert "path-200.txt" not in {row["path"] for row in rows}
    assert "205 total" in line
    assert sum(line.count(f"path-{index:03}.txt") for index in range(205)) == 5
