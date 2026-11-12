"""Security regressions for mutation edit commits."""

import errno
import hashlib
import os
import sys
from types import SimpleNamespace

import kernel_mutate_edits
import pytest
from kernel_mutate_edits import EditConflict, MutationEdits
from kernel_mutate_records import Journal


def test_write_seam_refuses_unbound_invocation(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("before")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError, match="parent changed"):
            kernel_mutate_edits._write_at(parent_fd, target.name, b"after", 0o644)
    finally:
        os.close(parent_fd)
    assert target.read_text() == "before"


@pytest.mark.parametrize(("move_at", "replace_path"), [
    ("prepared", False), ("prepared", True), ("temporary_sync", False),
])
def test_edit_refuses_parent_relocated_after_validation(tmp_path, monkeypatch, move_at, replace_path):
    repo = tmp_path / "workspace"
    parent = repo / "allowed"
    parent.mkdir(parents=True)
    target = parent / "data.txt"
    target.write_text("before")
    relocated = tmp_path / "relocated"
    journal = Journal(tmp_path / "journal")
    runtime = SimpleNamespace(
        repo=repo, journal=journal, effect_row=None, edit_scope=None,
        protected_paths=set(),
        policy=SimpleNamespace(write_globs=("allowed/*.txt",)),
    )
    edits = MutationEdits(runtime)
    staged = edits.stage("relocated-parent", [{
        "op": "replace", "path": "allowed/data.txt",
        "base_sha256": hashlib.sha256(b"before").hexdigest(), "text": "after",
    }])
    append = journal.append

    def relocate():
        parent.rename(relocated)
        if replace_path:
            parent.mkdir()
            target.write_text("decoy")

    def relocate_after_validation(kind, **fields):
        result = append(kind, **fields)
        if kind == "EFFECT_PREPARED" and move_at == "prepared":
            relocate()
        return result

    monkeypatch.setattr(journal, "append", relocate_after_validation)
    if move_at == "temporary_sync":
        chmod = os.fchmod

        def relocate_after_temporary_sync(fd, mode):
            result = chmod(fd, mode)
            relocate()
            return result

        monkeypatch.setattr(os, "fchmod", relocate_after_temporary_sync)
    with pytest.raises(EditConflict, match="parent"):
        edits.commit(staged, cleanup_ok=True)

    assert (relocated / "data.txt").read_text() == "before"
    if replace_path:
        assert target.read_text() == "decoy"
    else:
        assert not target.exists()
    assert not list(relocated.glob(".sf-mutate-*"))
    assert any(row["kind"] == "EFFECT_REFUSED" for row in journal.events())


def test_edit_recovery_refuses_parent_relocated_after_validation(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    parent = repo / "allowed"
    parent.mkdir(parents=True)
    target = parent / "data.txt"
    target.write_text("before")
    relocated = tmp_path / "relocated"
    journal = Journal(tmp_path / "journal")
    runtime = SimpleNamespace(
        repo=repo, journal=journal, effect_row=None, edit_scope=None,
        protected_paths=set(), policy=SimpleNamespace(write_globs=("allowed/*.txt",)),
    )
    edits = MutationEdits(runtime)
    staged = edits.stage("recover-relocated-parent", [{
        "op": "replace", "path": "allowed/data.txt",
        "base_sha256": hashlib.sha256(b"before").hexdigest(), "text": "after",
    }])
    before_blob = journal.put_blob(b"before")
    after_blob = journal.put_blob(b"after")
    journal.append(
        "EFFECT_PREPARED", call=staged.call,
        files=[{"path": "allowed/data.txt", "pre_sha256": before_blob,
                "post_sha256": after_blob, "pre_blob": before_blob, "post_blob": after_blob}],
        root_identity=list(staged.root_identity),
        parent_identities=[list(staged.files[0]["parent_identity"])],
    )
    read_at = kernel_mutate_edits._read_at
    first_read = True

    def relocate_after_recovery_validation(parent_fd, name):
        nonlocal first_read
        result = read_at(parent_fd, name)
        if first_read:
            first_read = False
            parent.rename(relocated)
        return result

    monkeypatch.setattr(kernel_mutate_edits, "_read_at", relocate_after_recovery_validation)
    with pytest.raises(EditConflict, match="parent"):
        edits.recover(staged.call)

    assert (relocated / "data.txt").read_text() == "before"
    assert not target.exists()
    assert not any(row["kind"] == "EFFECT_COMMITTED" for row in journal.events())


@pytest.mark.parametrize("operation", ("replace", "create", "delete"))
def test_edit_rolls_back_parent_relocated_at_rename(tmp_path, monkeypatch, operation):
    repo = tmp_path / "workspace"
    parent = repo / "allowed"
    parent.mkdir(parents=True)
    target = parent / "data.txt"
    if operation != "create":
        target.write_text("before")
    relocated = tmp_path / "relocated"
    journal = Journal(tmp_path / "journal")
    runtime = SimpleNamespace(
        repo=repo, journal=journal, effect_row=None, edit_scope=None,
        protected_paths=set(), policy=SimpleNamespace(write_globs=("allowed/*.txt",)),
    )
    edits = MutationEdits(runtime)
    proposal = {"op": operation, "path": "allowed/data.txt",
                "base_sha256": None if operation == "create" else hashlib.sha256(b"before").hexdigest()}
    if operation != "delete":
        proposal["text"] = "after"
    staged = edits.stage("rename-race", [proposal])
    moved = False

    def relocate_at_operation():
        nonlocal moved
        if not moved:
            moved = True
            parent.rename(relocated)

    monkeypatch.setattr(kernel_mutate_edits, "_before_edit_operation", relocate_at_operation)
    with pytest.raises(EditConflict):
        edits.commit(staged, cleanup_ok=True)

    assert moved
    if operation == "create":
        assert not (relocated / "data.txt").exists()
    else:
        assert (relocated / "data.txt").read_text() == "before"
    assert not list(relocated.glob(".sf-mutate-*"))
    assert not target.exists()


@pytest.mark.parametrize("interfere", (False, True))
def test_edit_rolls_back_prior_file_when_parent_moves_on_second_operation(tmp_path, monkeypatch,
                                                                        interfere):
    repo = tmp_path / "workspace"
    parent = repo / "allowed"
    parent.mkdir(parents=True)
    for name in ("a.txt", "b.txt"):
        (parent / name).write_text("before")
    relocated = tmp_path / "relocated"
    journal = Journal(tmp_path / "journal")
    runtime = SimpleNamespace(
        repo=repo, journal=journal, effect_row=None, edit_scope=None,
        protected_paths=set(), policy=SimpleNamespace(write_globs=("allowed/*.txt",)),
    )
    edits = MutationEdits(runtime)
    staged = edits.stage("second-rename-race", [
        {"op": "replace", "path": f"allowed/{name}",
         "base_sha256": hashlib.sha256(b"before").hexdigest(), "text": "after"}
        for name in ("a.txt", "b.txt")
    ])
    identities = [(parent / name).stat().st_ino for name in ("a.txt", "b.txt")]
    rename = kernel_mutate_edits._rename_at2
    operations = 0

    def relocate_on_second_operation(src_fd, src, dst_fd, dst, flags):
        nonlocal operations
        operations += 1
        if operations == 2:
            parent.rename(relocated)
        result = rename(src_fd, src, dst_fd, dst, flags)
        if operations == 2 and interfere:
            (relocated / "b.txt").write_text("host")
        return result

    monkeypatch.setattr(kernel_mutate_edits, "_rename_at2", relocate_on_second_operation)
    with pytest.raises(EditConflict):
        edits.commit(staged, cleanup_ok=True)

    assert operations == (3 if interfere else 4)
    assert (relocated / "a.txt").read_text() == "before"
    assert (relocated / "a.txt").stat().st_ino == identities[0]
    if interfere:
        assert (relocated / "b.txt").read_text() == "host"
        assert any(row["kind"] == "EFFECT_INDETERMINATE" for row in journal.events())
        assert [path.read_text() for path in relocated.glob(".sf-mutate-*")] == ["before"]
    else:
        assert (relocated / "b.txt").read_text() == "before"
        assert (relocated / "b.txt").stat().st_ino == identities[1]
        assert not list(relocated.glob(".sf-mutate-*"))


def test_create_does_not_replace_target_added_at_operation(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    parent = repo / "allowed"
    parent.mkdir(parents=True)
    target = parent / "data.txt"
    journal = Journal(tmp_path / "journal")
    runtime = SimpleNamespace(
        repo=repo, journal=journal, effect_row=None, edit_scope=None,
        protected_paths=set(), policy=SimpleNamespace(write_globs=("allowed/*.txt",)),
    )
    edits = MutationEdits(runtime)
    staged = edits.stage("create-race", [
        {"op": "create", "path": "allowed/data.txt", "base_sha256": None, "text": "edit"},
    ])
    monkeypatch.setattr(kernel_mutate_edits, "_before_edit_operation",
                        lambda: target.write_text("host"))

    with pytest.raises(EditConflict):
        edits.commit(staged, cleanup_ok=True)

    assert target.read_text() == "host"
    assert not list(parent.glob(".sf-mutate-*"))


@pytest.mark.parametrize("operation", ("replace", "create", "delete"))
@pytest.mark.parametrize("interfere", (False, "bytes", "inode"))
@pytest.mark.parametrize("recover", (False, True))
def test_syscall_boundary_relocation_validates_compensation(tmp_path, monkeypatch,
                                                           operation, interfere, recover):
    repo = tmp_path / "workspace"
    parent = repo / "allowed"
    parent.mkdir(parents=True)
    target = parent / "data.txt"
    if operation != "create":
        target.write_text("before")
    relocated = tmp_path / "relocated"
    journal = Journal(tmp_path / "journal")
    edits = MutationEdits(SimpleNamespace(
        repo=repo, journal=journal, effect_row=None, edit_scope=None,
        protected_paths=set(), policy=SimpleNamespace(write_globs=("allowed/*.txt",)),
    ))
    proposal = {"op": operation, "path": "allowed/data.txt",
                "base_sha256": None if operation == "create" else hashlib.sha256(b"before").hexdigest()}
    if operation != "delete":
        proposal["text"] = "after"
    staged = edits.stage("syscall-race", [proposal])
    if recover:
        pre = journal.put_blob(b"before") if operation != "create" else None
        post = journal.put_blob(b"after") if operation != "delete" else None
        journal.append("EFFECT_PREPARED", call=staged.call,
                       files=[{"path": "allowed/data.txt", "pre_sha256": pre,
                               "post_sha256": post, "pre_blob": pre, "post_blob": post}],
                       root_identity=list(staged.root_identity),
                       parent_identities=[list(staged.files[0]["parent_identity"])])
    host_content = "after" if interfere == "inode" and operation != "delete" else "host"
    rename = kernel_mutate_edits._rename_at2
    calls = 0
    transient = []

    def race(src_fd, src, dst_fd, dst, flags):
        nonlocal calls
        calls += 1
        if calls == 1:
            # Exactly after the caller's final validation, before the actual syscall.
            parent.rename(relocated)
            parent.mkdir()
            target.write_text("decoy")
        result = rename(src_fd, src, dst_fd, dst, flags)
        if calls == 1:
            moved_target = relocated / "data.txt"
            transient.append(moved_target.read_text() if moved_target.exists() else None)
            if interfere:
                # A host entry must never be overwritten by an unchecked inverse.
                if moved_target.exists():
                    moved_target.rename(relocated / "saved-by-host")
                moved_target.write_text(host_content)
        return result

    monkeypatch.setattr(kernel_mutate_edits, "_rename_at2", race)
    with pytest.raises(EditConflict):
        if recover:
            edits.recover(staged.call)
        else:
            edits.commit(staged, cleanup_ok=True)
    assert transient == [None if operation == "delete" else "after"]
    assert target.read_text() == "decoy"
    moved_target = relocated / "data.txt"
    if interfere:
        assert moved_target.read_text() == host_content
        assert any(row["kind"] == "EFFECT_INDETERMINATE" for row in journal.events())
    else:
        assert (moved_target.read_text() if moved_target.exists() else None) == (
            None if operation == "create" else "before")
        assert not list(relocated.glob(".sf-mutate-*"))
        assert any(row["kind"] == "EFFECT_REFUSED" for row in journal.events())
    assert not any(row["kind"] == "EFFECT_COMMITTED" for row in journal.events())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux openat2 contract")
@pytest.mark.parametrize("lookup_errno", (errno.ENOSYS, errno.EAGAIN, errno.EXDEV, errno.ELOOP))
def test_constrained_parent_lookup_fails_closed(tmp_path, monkeypatch, lookup_errno):
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)

    def syscall(*args):
        kernel_mutate_edits.ctypes.set_errno(lookup_errno)
        return -1

    monkeypatch.setattr(kernel_mutate_edits.ctypes, "CDLL",
                        lambda *args, **kwargs: SimpleNamespace(syscall=syscall))
    try:
        with pytest.raises(EditConflict, match="constrained lookup failed"):
            kernel_mutate_edits._parent(root_fd, ("allowed", "data.txt"))
    finally:
        os.close(root_fd)
