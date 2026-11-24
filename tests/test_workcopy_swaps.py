"""A path that changes between file and directory reaches the private working copy."""
from pathlib import Path

import pytest

from stateframe import workcopy


def _work(tmp_path: Path, files: dict[str, str]) -> workcopy.WorkCopy:
    repo = tmp_path / "repo"
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="ascii")
    return workcopy.create(repo, tmp_path / "copy")


def test_file_becomes_directory(tmp_path):
    work = _work(tmp_path, {"keep.py": "k", "pkg/mod.py": "old"})
    files = {"pkg/mod.py": None, "pkg/mod.py/deep/child.py": b"new"}
    workcopy.apply_files(work, files, declared=set(files))
    assert (work.copy / "pkg/mod.py/deep/child.py").read_bytes() == b"new"
    assert workcopy.apply_files(work, files, declared=set(files)) == []


def test_directory_becomes_file(tmp_path):
    work = _work(tmp_path, {"keep.py": "k", "pkg/a/one.py": "1", "pkg/a/b/two.py": "2"})
    files = {"pkg/a": b"new", "pkg/a/one.py": None, "pkg/a/b/two.py": None}
    workcopy.apply_files(work, files, declared=set(files))
    assert (work.copy / "pkg/a").read_bytes() == b"new"
    assert workcopy.apply_files(work, files, declared=set(files)) == []


def test_undeclared_child_keeps_the_directory(tmp_path):
    work = _work(tmp_path, {"keep.py": "k", "pkg/a/one.py": "1", "pkg/a/two.py": "2"})
    files = {"pkg/a": b"new", "pkg/a/one.py": None, "pkg/a/two.py": None}
    with pytest.raises(workcopy.WorkCopyError):
        workcopy.apply_files(work, files, declared={"pkg/a", "pkg/a/one.py"})
    assert (work.copy / "pkg/a/two.py").read_text() == "2"


def test_symlinked_directory_is_not_entered(tmp_path):
    work = _work(tmp_path, {"keep.py": "k", "pkg/one.py": "1"})
    outside = tmp_path / "outside"
    outside.mkdir()
    (work.copy / "link").symlink_to(outside, target_is_directory=True)
    files = {"link": b"new", "link/x.py": None}
    with pytest.raises(workcopy.WorkCopyError):
        workcopy.apply_files(work, files, declared=set(files))
    assert outside.is_dir() and (work.copy / "link").is_symlink()


@pytest.mark.parametrize("to_file", [True, False])
def test_in_place_swap_refuses_and_leaves_the_original(tmp_path, to_file):
    old = {"keep.py": "k", "pkg/a/one.py": "1"} if to_file else {"keep.py": "k", "pkg/a": "1"}
    work = _work(tmp_path, old)
    files = ({"pkg/a": b"new", "pkg/a/one.py": None} if to_file
             else {"pkg/a": None, "pkg/a/one.py": b"new"})
    workcopy.apply_files(work, files, declared=set(files))
    with pytest.raises(workcopy.WorkCopyError):
        workcopy.apply_in_place(work)
    assert {rel: (work.original / rel).read_text() for rel in old} == old


def test_undeclared_new_file_keeps_the_directory(tmp_path):
    work = _work(tmp_path, {"keep.py": "k", "pkg/a/one.py": "1"})
    (work.copy / "pkg/a/stray.py").write_text("s")
    files = {"pkg/a": b"new", "pkg/a/one.py": None, "pkg/a/stray.py": None}
    with pytest.raises(workcopy.WorkCopyError):
        workcopy.apply_files(work, files, declared={"pkg/a", "pkg/a/one.py"})
    assert (work.copy / "pkg/a/stray.py").read_text() == "s"
