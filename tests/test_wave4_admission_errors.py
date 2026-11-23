from __future__ import annotations

import errno
import os
from pathlib import Path

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
    else:
        assert legacy[0] == "admitted"
        assert set(legacy[1]) == {"sample.py", "nested/component/module.py"}
        assert legacy[2] == ["ignored.pyc", "nested/component/loop"]
    assert observe(True) == legacy
