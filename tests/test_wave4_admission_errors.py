from __future__ import annotations

import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import workcopy
from stateframe.workspace_start_snapshot import pin_parent


@pytest.mark.parametrize("case, operation, error_number", [
    ("unreadable-directory", "scan", errno.EACCES),
    ("unreadable-file", "read", errno.EACCES),
    ("removed-directory", "scan", errno.ENOENT),
    ("symlink-loop", None, None),
    ("permissions-changed", "scan", errno.EACCES),
    ("interrupted-scan", "scan", errno.EIO),
])
def test_wave4_matches_legacy_admission_errors(tmp_path, monkeypatch, case, operation, error_number):
    source = tmp_path / "source"
    component = source / "nested" / "component"
    component.mkdir(parents=True)
    module = component / "module.py"
    module.write_text("MODULE = True\n")
    (source / "sample.py").write_text("SAMPLE = True\n")
    (source / "ignored.pyc").write_bytes(b"cache")
    if case == "symlink-loop":
        (component / "loop").symlink_to("loop")
        (component / "dir-link").symlink_to("../..", target_is_directory=True)

    victim = module if case == "unreadable-file" else component
    mode = victim.stat().st_mode & 0o777
    if error_number == errno.EACCES:
        victim.chmod(0)
        try:
            if os.access(victim, os.R_OK):
                pytest.skip("this account can read files and directories without read permission")
        finally:
            victim.chmod(mode)

    real_ignored = workcopy._ignored
    real_scandir = os.scandir
    workcopy_type = workcopy.WorkCopy
    component_inode = component.stat().st_ino
    aside = tmp_path / "removed"

    def ignored(name, is_dir):
        # The nested parent has already listed component, but the walk has yet to enter it.
        if is_dir and name == component.name:
            if case == "removed-directory":
                component.rename(aside)
            elif case == "permissions-changed":
                component.chmod(0)
        return real_ignored(name, is_dir)

    def scandir(path):
        matches = (os.fstat(path).st_ino == component_inode if isinstance(path, int)
                   else Path(path) == component)
        if case == "interrupted-scan" and matches:
            raise OSError(errno.EIO, os.strerror(errno.EIO), path)
        return real_scandir(path)

    def observe(admitted):
        work = None

        def construct(**kwargs):
            nonlocal work
            work = workcopy_type(**kwargs)
            return work

        run = tmp_path / ("on" if admitted else "off")
        with monkeypatch.context() as patch:
            patch.setattr(workcopy, "WorkCopy", construct)
            patch.setattr(workcopy, "_ignored", ignored)
            # The space check walks first; these faults belong to the copy.
            patch.setattr(workcopy, "_check_copy_space", lambda *_args, **_kwargs: False)
            patch.setattr(os, "scandir", scandir)
            if case in {"unreadable-directory", "unreadable-file"}:
                victim.chmod(0)
            try:
                workcopy.create(source, run / "workspace", admit_snapshot=admitted)
            except workcopy.WorkCopyScanError as error:
                result = ("refused", str(error))
            else:
                result = ("admitted", work.baseline)
                if admitted:
                    pin_parent(work, run / "parent")
            finally:
                if aside.exists():
                    aside.rename(component)
                victim.chmod(mode)
        if admitted and result[0] == "refused":
            assert list(run.iterdir()) == []
        return (*result, work.skipped)

    legacy = observe(False)
    if operation == "scan":
        reason = f"cannot scan repository directory {victim}: {os.strerror(error_number)}"
        assert legacy == ("refused", reason, ["ignored.pyc"])
    elif operation == "read":
        reason = (f"cannot read repository file {victim}: "
                  f"[Errno {error_number}] {os.strerror(error_number)}: '{victim}'")
        assert legacy == ("refused", reason, ["ignored.pyc"])
        admitted = observe(True)
        assert admitted == ("admitted", {"sample.py": workcopy._sha(b"SAMPLE = True\n")},
                           ["ignored.pyc", "nested/component/module.py"])
    else:
        assert legacy[0] == "admitted"
        assert set(legacy[1]) == {"sample.py", "nested/component/module.py"}
        assert legacy[2] == ["ignored.pyc", "nested/component/loop"]
        assert observe(True) == legacy


def test_eager_copy_reports_an_unreadable_file_by_name(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    blocked = source / "blocked.py"
    blocked.write_text("value = 1\n")
    real_open = os.open

    def deny(path, flags, *args, **kwargs):
        if Path(path).name == blocked.name:
            raise PermissionError(errno.EACCES, "permission denied", os.fspath(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny)
    with pytest.raises(workcopy.WorkCopyScanError, match=str(blocked)) as error:
        workcopy.create(source, tmp_path / "run" / "workspace", admit_snapshot=False)
    assert "cannot read repository file" in str(error.value)


def test_eager_copy_reports_an_unreadable_directory_by_name(tmp_path, monkeypatch):
    source = tmp_path / "source"
    blocked = source / "blocked"
    blocked.mkdir(parents=True)
    (blocked / "inside.py").write_text("value = 1\n")
    real_scandir = os.scandir

    def deny(path):
        if isinstance(path, int) and os.fstat(path).st_ino == blocked.stat().st_ino:
            raise PermissionError(errno.EACCES, "permission denied", str(blocked))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", deny)
    with pytest.raises(workcopy.WorkCopyScanError, match=str(blocked)):
        workcopy.create(source, tmp_path / "run" / "workspace", admit_snapshot=False)


def test_eager_copy_keeps_legacy_skips_and_directory_modes(tmp_path):
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "keep.py").write_text("value = 1\n")
    (nested / "inside.py").write_text("value = 1\n")
    (source / "link.py").symlink_to("keep.py")
    os.mkfifo(source / "pipe")
    reference = tmp_path / "reference"
    reference.mkdir()
    expected_mode = reference.stat().st_mode & 0o777
    source_mode = 0o700 if expected_mode != 0o700 else 0o755
    nested.chmod(source_mode)

    work = workcopy.create(source, tmp_path / "run" / "workspace", admit_snapshot=False)

    assert work.skipped == ["link.py", "pipe"]
    assert not (work.copy / "link.py").exists()
    assert not (work.copy / "pipe").exists()
    assert (work.copy / "nested").stat().st_mode & 0o777 == expected_mode
    assert expected_mode != source_mode


def test_eager_copy_checks_space_before_writing_streamed_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    with (source / "large.bin").open("wb") as stream:
        stream.truncate((65 << 20))
    destination = tmp_path / "run" / "workspace"
    monkeypatch.setattr(workcopy, "_reflink",
                        lambda *_: (_ for _ in ()).throw(OSError(errno.EIO, "clone unavailable")))
    free = [1 << 40, 0]

    def statvfs(_path):
        value = free.pop(0) if len(free) > 1 else free[0]
        return SimpleNamespace(f_bavail=value, f_frsize=1, f_blocks=1 << 40)

    monkeypatch.setattr(workcopy.os, "statvfs", statvfs, raising=False)
    add = workcopy._CopySpace.add
    observed = []

    def before_write(space, count):
        observed.append((count, (destination / "large.bin").stat().st_size))
        return add(space, count)

    monkeypatch.setattr(workcopy._CopySpace, "add", before_write)
    with pytest.raises(workcopy.WorkCopyError, match="--repo"):
        workcopy.create(source, destination, admit_snapshot=False)
    assert observed == [((65 << 20), 0)]
