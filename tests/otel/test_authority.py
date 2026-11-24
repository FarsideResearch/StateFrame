from __future__ import annotations

from types import SimpleNamespace

import os
from pathlib import Path
import stat

from stateframe.otel import authority


def _private_group_lookups(monkeypatch, *, members=(), primary_users=(), fail=None):
    username = "current-user"
    gid = os.getgid()
    user = SimpleNamespace(pw_name=username, pw_uid=os.getuid(), pw_gid=gid, pw_dir="/home/current-user")
    monkeypatch.setattr(authority.pwd, "getpwuid", lambda _uid: user)
    monkeypatch.setattr(
        authority.pwd,
        "getpwall",
        lambda: [user, *(SimpleNamespace(pw_name=name, pw_uid=index + 100_000,
                                         pw_gid=gid, pw_dir="/home/other")
                         for index, name in enumerate(primary_users))],
    )

    def getgrgid(_gid):
        if fail:
            raise fail
        return SimpleNamespace(gr_gid=gid, gr_mem=list(members))

    monkeypatch.setattr(authority, "grp", SimpleNamespace(getgrgid=getgrgid), raising=False)
    return gid


def _secure_other_ancestors(monkeypatch, parent: Path, run: Path, gid: int):
    original_lstat = Path.lstat

    def controlled_lstat(path):
        info = original_lstat(path)
        mode = info.st_mode
        if path not in {parent, run}:
            mode &= ~stat.S_IWGRP
        return SimpleNamespace(st_mode=mode, st_uid=info.st_uid,
                               st_gid=gid if path in {parent, run} else info.st_gid)

    monkeypatch.setattr(Path, "lstat", controlled_lstat)


def test_group_writable_parent_is_allowed_for_current_users_private_group(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o775)
    run = parent / "run"
    run.mkdir(mode=0o700)
    gid = _private_group_lookups(monkeypatch)
    _secure_other_ancestors(monkeypatch, parent, run, gid)

    assert authority.source_mode_problem(run) is None
    assert authority._private_root(run)


def test_group_writable_parent_requires_a_private_group(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o775)
    run = parent / "run"
    run.mkdir(mode=0o700)

    gid = _private_group_lookups(monkeypatch, members=("other-user",))
    _secure_other_ancestors(monkeypatch, parent, run, gid)
    assert authority.source_mode_problem(run) == (parent, "go-w")
    assert not authority._private_root(run)

    gid = _private_group_lookups(monkeypatch, primary_users=("other-user",))
    _secure_other_ancestors(monkeypatch, parent, run, gid)
    assert authority.source_mode_problem(run) == (parent, "go-w")

    gid = _private_group_lookups(monkeypatch, fail=KeyError("missing group"))
    _secure_other_ancestors(monkeypatch, parent, run, gid)
    assert authority.source_mode_problem(run) == (parent, "go-w")


def test_group_writable_run_directory_keeps_owner_only_rule(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o775)
    run = parent / "run"
    run.mkdir(mode=0o775)
    run.chmod(0o775)
    gid = _private_group_lookups(monkeypatch)
    _secure_other_ancestors(monkeypatch, parent, run, gid)

    assert authority.source_mode_problem(run) == (run, "700")


def test_world_writable_parent_remains_refused_with_private_group(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    parent.chmod(0o777)
    run = parent / "run"
    run.mkdir(mode=0o700)
    gid = _private_group_lookups(monkeypatch)
    _secure_other_ancestors(monkeypatch, parent, run, gid)

    assert authority.source_mode_problem(run) == (parent, "go-w")
    assert not authority._private_root(run)
