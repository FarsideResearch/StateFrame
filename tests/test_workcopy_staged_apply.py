"""Regressions for bounded descriptors and identity-safe staged promotion."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import resource
import stat

import pytest

from stateframe import workcopy


def _baseline(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_six_hundred_changed_files_fit_under_256_descriptors(tmp_path: Path) -> None:
    if not hasattr(resource, "RLIMIT_NOFILE"):
        pytest.skip("RLIMIT_NOFILE is unavailable")
    root = tmp_path / "repo"
    root.mkdir()
    before = {f"file-{index:03}.txt": b"before\n" for index in range(600)}
    for rel, data in before.items():
        (root / rel).write_bytes(data)
    changes = {rel: b"after\n" for rel in before}
    baseline = {rel: _baseline(data) for rel, data in before.items()}

    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if old_hard < 256:
        pytest.skip("host descriptor hard limit is below 256")
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, old_hard))
        applied = workcopy.guarded_apply(root, changes, baseline, what="test apply")
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (old_soft, old_hard))

    assert len(applied) == 600
    assert all((root / rel).read_bytes() == b"after\n" for rel in before)


def test_parent_swap_after_staging_is_refused_and_temp_is_removed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    parent = root / "pkg"
    parent.mkdir(parents=True)
    rel = "pkg/target.py"
    original = b"before\n"
    (parent / "target.py").write_bytes(original)
    moved = root / "pkg-moved"
    original_read_at = workcopy._read_at
    reads = 0

    def swap_parent_after_final_read(descriptor: int, name: str):
        nonlocal reads
        result = original_read_at(descriptor, name)
        if name == "target.py":
            reads += 1
            if reads == 2:
                parent.rename(moved)
                parent.mkdir()
                (parent / "target.py").write_bytes(b"replacement user bytes\n")
        return result

    monkeypatch.setattr(workcopy, "_read_at", swap_parent_after_final_read)
    with pytest.raises(workcopy.WorkCopyError, match="parent directory changed while applying"):
        workcopy.guarded_apply(root, {rel: b"candidate\n"}, {rel: _baseline(original)},
                               what="test apply")

    assert reads == 2
    assert (moved / "target.py").read_bytes() == original
    assert (parent / "target.py").read_bytes() == b"replacement user bytes\n"
    assert not list(root.rglob(".stateframe-*.tmp"))


def test_staged_temp_swap_is_refused_and_replacement_temp_is_preserved(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    rel = "target.py"
    original = b"before\n"
    (root / rel).write_bytes(original)
    original_read_at = workcopy._read_at
    reads = 0

    def swap_temp_after_final_read(descriptor: int, name: str):
        nonlocal reads
        result = original_read_at(descriptor, name)
        if name == rel:
            reads += 1
            if reads == 3:
                temp = next(root.glob(".stateframe-*.tmp"))
                replacement = root / ".replacement"
                replacement.write_bytes(b"replacement temporary\n")
                os.replace(replacement, temp)
        return result

    monkeypatch.setattr(workcopy, "_read_at", swap_temp_after_final_read)
    with pytest.raises(workcopy.WorkCopyError, match="staged temporary changed while applying"):
        workcopy.guarded_apply(root, {rel: b"candidate\n"}, {rel: _baseline(original)},
                               what="test apply")

    assert reads == 3
    assert (root / rel).read_bytes() == original
    temps = list(root.glob(".stateframe-*.tmp"))
    assert len(temps) == 1
    assert temps[0].read_bytes() == b"replacement temporary\n"


@pytest.mark.parametrize("keep_nonempty", [False, True], ids=["empty-created-dir", "nonempty-created-dir"])
def test_failed_staging_cleans_empty_created_dirs_and_keeps_protected_dirs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_nonempty: bool) -> None:
    root = tmp_path / "repo"
    existing = root / "existing"
    existing.mkdir(parents=True)

    def fail_after_temp_create(descriptor: int, data: bytes) -> int:
        if keep_nonempty:
            (existing / "new" / "keep.txt").write_bytes(b"outside content\n")
        raise OSError("injected staging failure")

    monkeypatch.setattr(workcopy.os, "write", fail_after_temp_create)
    with pytest.raises(workcopy.WorkCopyError, match="staging failed"):
        workcopy.guarded_apply(root, {"existing/new/target.py": b"candidate\n"}, {},
                               what="test apply")

    assert existing.is_dir()
    assert not list(root.rglob(".stateframe-*.tmp"))
    if keep_nonempty:
        assert (existing / "new" / "keep.txt").read_bytes() == b"outside content\n"
    else:
        assert not (root / "existing" / "new").exists()


def test_temp_is_removed_when_identity_capture_fails_after_creation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    original_open = os.open
    original_fstat = os.fstat
    staged_fds: set[int] = set()
    failed_fds: set[int] = set()

    def tracking_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if isinstance(path, str) and path.startswith(".stateframe-") and path.endswith(".tmp"):
            staged_fds.add(descriptor)
        return descriptor

    def fail_temp_fstat(descriptor: int):
        if descriptor in staged_fds and descriptor not in failed_fds:
            failed_fds.add(descriptor)
            raise OSError("injected staged identity failure")
        return original_fstat(descriptor)

    monkeypatch.setattr(workcopy.os, "open", tracking_open)
    monkeypatch.setattr(workcopy.os, "fstat", fail_temp_fstat)
    with pytest.raises(workcopy.WorkCopyError, match="staging failed"):
        workcopy.guarded_apply(root, {"new/target.py": b"candidate\n"}, {}, what="test apply")

    assert failed_fds
    assert not list(root.rglob(".stateframe-*.tmp"))
    assert not (root / "new").exists()


def test_destination_edit_during_staged_chmod_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target.py"
    before = b"before\n"
    outside = b"outside edit\n"
    target.write_bytes(before)
    read_at, fchmod = workcopy._read_at, os.fchmod
    reads = 0
    edited = False

    def count_reads(parent: int, name: str):
        nonlocal reads
        reads += 1
        return read_at(parent, name)

    def edit_during_chmod(descriptor: int, mode: int) -> None:
        nonlocal edited
        # Inject during staging if the mode is set there, or during the
        # per-file replacement if apply still adjusts the mode there.
        if os.fstat(descriptor).st_size == 0 or reads >= 3:
            target.write_bytes(outside)
            edited = True
        fchmod(descriptor, mode)

    monkeypatch.setattr(workcopy, "_read_at", count_reads)
    monkeypatch.setattr(workcopy.os, "fchmod", edit_during_chmod)
    with pytest.raises(workcopy.WorkCopyError, match="changed in the original"):
        workcopy.guarded_apply(tmp_path, {target.name: b"candidate\n"},
                               {target.name: _baseline(before)}, what="test apply")

    assert edited
    assert target.read_bytes() == outside
    assert not list(tmp_path.glob(".stateframe-*.tmp"))


@pytest.mark.parametrize("existing", [True, False], ids=["existing", "new"])
def test_staged_contents_have_destination_mode_when_first_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool) -> None:
    target = tmp_path / "target.py"
    baseline = {}
    if existing:
        target.write_bytes(b"before\n")
        target.chmod(0o600)
        baseline[target.name] = _baseline(b"before\n")
    creation_modes = []
    written_modes = []
    original_open, original_write = os.open, os.write

    def capture_creation_mode(path, flags, mode=0o777, **kwargs):
        descriptor = original_open(path, flags, mode, **kwargs)
        if Path(path).match(".stateframe-*.tmp") and flags & os.O_CREAT:
            creation_modes.append((mode, stat.S_IMODE(os.fstat(descriptor).st_mode)))
        return descriptor

    def capture_written_mode(descriptor: int, data: bytes) -> int:
        count = original_write(descriptor, data)
        written_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        return count

    old_umask = os.umask(0o027)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(workcopy.os, "open", capture_creation_mode)
            patch.setattr(workcopy.os, "write", capture_written_mode)
            assert workcopy.guarded_apply(tmp_path, {target.name: b"candidate\n"}, baseline,
                                          what="test apply") == [target.name]
    finally:
        os.umask(old_umask)

    expected_mode = 0o600 if existing else 0o640
    assert creation_modes == [(0o600, 0o600)]
    assert written_modes == [expected_mode]
    assert stat.S_IMODE(target.stat().st_mode) == expected_mode


def test_failed_unlink_in_displaced_parent_leaves_other_temps_cleaned_and_fds_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "pkg"
    parent.mkdir()
    moved = tmp_path / "moved"
    before = b"before\n"
    paths = ["pkg/a.py", "pkg/b.py"]
    for rel in paths:
        (tmp_path / rel).write_bytes(before)

    original_open, original_unlink = os.open, os.unlink
    opened_fds = set()
    blocked_temp = None

    def record_open(*args, **kwargs):
        descriptor = original_open(*args, **kwargs)
        opened_fds.add(descriptor)
        return descriptor

    def move_then_fail(*args, **kwargs):
        parent.rename(moved)
        raise OSError("injected replace failure")

    def fail_one_unlink(path, **kwargs):
        nonlocal blocked_temp
        if Path(path).match(".stateframe-*.tmp"):
            if blocked_temp is None:
                blocked_temp = path
            if path == blocked_temp:
                raise PermissionError("injected temporary unlink failure")
        original_unlink(path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "open", record_open)
        patch.setattr(workcopy.os, "replace", move_then_fail)
        patch.setattr(workcopy.os, "unlink", fail_one_unlink)
        with pytest.raises(workcopy.WorkCopyError) as caught:
            workcopy.guarded_apply(tmp_path, dict.fromkeys(paths, b"candidate\n"),
                                   dict.fromkeys(paths, _baseline(before)), what="test apply")

    for descriptor in opened_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert blocked_temp is not None
    assert list(moved.glob(".stateframe-*.tmp")) == [moved / blocked_temp]
    assert all((moved / Path(rel).name).read_bytes() == before for rel in paths)
    assert "injected replace failure" in str(caught.value)


def test_destination_mode_tightened_after_preflight_prevents_wider_staging_write(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target.py"
    before = b"before\n"
    target.write_bytes(before)
    target.chmod(0o644)
    read_at, write = workcopy._read_at, os.write
    tightened = False
    staged_modes = []

    def tighten_after_preflight(parent: int, name: str):
        nonlocal tightened
        result = read_at(parent, name)
        if not tightened:
            target.chmod(0o600)
            tightened = True
        return result

    def record_mode(descriptor: int, data: bytes) -> int:
        staged_modes.append(stat.S_IMODE(os.fstat(descriptor).st_mode))
        return write(descriptor, data)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy, "_read_at", tighten_after_preflight)
        patch.setattr(workcopy.os, "write", record_mode)
        with pytest.raises(workcopy.WorkCopyError, match="destination mode changed"):
            workcopy.guarded_apply(tmp_path, {target.name: b"candidate\n"},
                                   {target.name: _baseline(before)}, what="test apply")

    assert tightened
    assert staged_modes == []
    assert target.read_bytes() == before
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".stateframe-*"))


def test_mode_probe_identity_failure_cleans_probe_and_closes_fd(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    open_file, fstat = os.open, os.fstat
    probe_fd = None

    def record_probe(path, *args, **kwargs):
        nonlocal probe_fd
        descriptor = open_file(path, *args, **kwargs)
        if Path(path).match(".stateframe-*.mode"):
            probe_fd = descriptor
        return descriptor

    def fail_probe_fstat(descriptor: int):
        if descriptor == probe_fd:
            raise OSError("injected mode probe stat failure")
        return fstat(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "open", record_probe)
        patch.setattr(workcopy.os, "fstat", fail_probe_fstat)
        with pytest.raises(workcopy.WorkCopyError, match="injected mode probe stat failure"):
            workcopy.guarded_apply(tmp_path, {"target.py": b"candidate\n"}, {}, what="test apply")

    assert probe_fd is not None
    with pytest.raises(OSError):
        os.fstat(probe_fd)
    assert not list(tmp_path.iterdir())


def test_failed_unlink_in_displaced_parent_still_cleans_nested_temps(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / "pkg"
    (parent / "nested").mkdir(parents=True)
    moved = tmp_path / "moved"
    paths = ["pkg/a.py", "pkg/nested/b.py"]
    before = b"before\n"
    for rel in paths:
        (tmp_path / rel).write_bytes(before)
    unlink, fstat = os.unlink, os.fstat
    parent_inode = parent.stat().st_ino

    def move_then_fail(*args, **kwargs):
        parent.rename(moved)
        raise OSError("injected replace failure")

    def fail_parent_unlink(path, *, dir_fd):
        if Path(path).match(".stateframe-*.tmp") and fstat(dir_fd).st_ino == parent_inode:
            raise PermissionError("injected parent unlink failure")
        unlink(path, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "replace", move_then_fail)
        patch.setattr(workcopy.os, "unlink", fail_parent_unlink)
        with pytest.raises(workcopy.WorkCopyError, match="injected replace failure"):
            workcopy.guarded_apply(tmp_path, dict.fromkeys(paths, b"candidate\n"),
                                   dict.fromkeys(paths, _baseline(before)), what="test apply")

    assert len(list(moved.glob(".stateframe-*.tmp"))) == 1
    assert not list((moved / "nested").glob(".stateframe-*.tmp"))
    assert (moved / "nested" / "b.py").read_bytes() == before


@pytest.mark.parametrize("phase", ["stage", "recheck", "replace"])
def test_failed_unlink_still_cleans_sibling_temps_after_parent_leaves_repo(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    root = tmp_path / "repo"
    parent = root / "pkg"
    parent.mkdir(parents=True)
    moved = tmp_path / "moved"
    before = b"before\n"
    paths = ["pkg/a.py", "pkg/b.py"]
    for rel in paths:
        (root / rel).write_bytes(before)

    open_file, read_at, write, unlink = os.open, workcopy._read_at, os.write, os.unlink
    open_fds: set[int] = set()
    temp_names: list[str] = []
    failed_unlinks: list[str] = []

    def record_open(path, flags, *args, **kwargs):
        descriptor = open_file(path, flags, *args, **kwargs)
        open_fds.add(descriptor)
        if flags & os.O_CREAT and Path(path).match(".stateframe-*.tmp"):
            temp_names.append(path)
        return descriptor

    def fail_after_moving_parent(*args, **kwargs):
        parent.rename(moved)
        raise OSError("injected failure after moving parent")

    def write_staged(descriptor, data):
        if len(temp_names) == 2:
            fail_after_moving_parent()
        return write(descriptor, data)

    def read_destination(descriptor, name):
        result = read_at(descriptor, name)
        if len(temp_names) == 2 and name == "a.py":
            parent.rename(moved)
        return result

    def fail_first_unlink(path, *, dir_fd):
        if temp_names and path == temp_names[0]:
            failed_unlinks.append(path)
            raise PermissionError("injected temporary unlink failure")
        unlink(path, dir_fd=dir_fd)

    with monkeypatch.context() as patch:
        patch.setattr(workcopy.os, "open", record_open)
        patch.setattr(workcopy.os, "unlink", fail_first_unlink)
        if phase == "stage":
            patch.setattr(workcopy.os, "write", write_staged)
        elif phase == "recheck":
            patch.setattr(workcopy, "_read_at", read_destination)
        else:
            patch.setattr(workcopy.os, "replace", fail_after_moving_parent)
        error = "parent directory changed" if phase == "recheck" else "injected failure after moving parent"
        with pytest.raises(workcopy.WorkCopyError, match=error):
            workcopy.guarded_apply(root, dict.fromkeys(paths, b"candidate\n"),
                                   dict.fromkeys(paths, _baseline(before)), what="test apply")

    for descriptor in open_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert all((moved / Path(rel).name).read_bytes() == before for rel in paths)
    assert list(moved.glob(".stateframe-*.tmp")) == [moved / temp_names[0]]
    assert failed_unlinks == [temp_names[0]]


def test_replacement_keeps_setid_mode_bits(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    before = b"before\n"
    target.write_bytes(before)
    target.chmod(0o4755)
    if stat.S_IMODE(target.stat().st_mode) != 0o4755:
        pytest.skip("filesystem does not support set-user-ID files")

    assert workcopy.guarded_apply(tmp_path, {target.name: b"candidate\n"},
                                  {target.name: _baseline(before)}, what="test apply") == [target.name]

    assert stat.S_IMODE(target.stat().st_mode) == 0o4755
