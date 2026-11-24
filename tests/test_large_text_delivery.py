"""An edited text file over 1 MiB is delivered like a smaller one."""
import pytest

from stateframe import workcopy


@pytest.mark.parametrize("snapshot", [False, True])
def test_changed_large_text_is_in_the_diff_and_applied(tmp_path, snapshot):
    from stateframe.workspace_start_snapshot import pin_parent

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("A = 1\n")
    lines = "".join(f"row {index:07d}\n" for index in range(120_000))
    assert len(lines) > 1 << 20
    (repo / "data.txt").write_text(lines)
    work = workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=snapshot)
    pin_parent(work, tmp_path / "run" / "parent")
    (work.copy / "data.txt").write_text(lines.replace("row 0000005\n", "row changed\n"))
    diff = workcopy.unified_diff(work)
    assert "-row 0000005" in diff and "+row changed" in diff
    assert workcopy.apply_in_place(work) == ["data.txt"]
    assert "row changed\n" in (repo / "data.txt").read_text()


def test_apply_never_stops_halfway_on_a_free_space_reserve(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(80):
        (repo / f"f{index:02}.txt").write_text("a")
    work = workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=False)
    for index in range(80):
        (work.copy / f"f{index:02}.txt").write_bytes(b"b" * (1 << 20))
    from types import SimpleNamespace
    monkeypatch.setattr(workcopy.os, "statvfs", lambda _path: SimpleNamespace(
        f_bavail=200 << 20, f_frsize=1, f_blocks=100 << 30), raising=False)
    assert len(workcopy.apply_in_place(work)) == 80


def test_apply_that_cannot_fit_changes_nothing(tmp_path, monkeypatch):
    from types import SimpleNamespace

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("a")
    work = workcopy.create(repo, tmp_path / "run" / "workspace", admit_snapshot=False)
    (work.copy / "a.txt").write_bytes(b"b" * 5000)
    monkeypatch.setattr(workcopy.os, "statvfs", lambda _path: SimpleNamespace(
        f_bavail=100, f_frsize=1, f_blocks=1000), raising=False)
    with pytest.raises(workcopy.WorkCopyError, match="needs 5000 bytes"):
        workcopy.apply_in_place(work)
    assert (repo / "a.txt").read_text() == "a"
    assert [path.name for path in repo.iterdir()] == ["a.txt"]


def test_copy_of_a_run_tree_keeps_what_the_first_copy_leaves_out(tmp_path):
    source = tmp_path / "session"
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__" / "x.pyc").write_bytes(b"1")
    (source / "app.py").write_text("A = 1\n")
    copied = workcopy.copy_tree_checked(source, tmp_path / "next")
    assert sorted(p.relative_to(copied).as_posix() for p in copied.rglob("*") if p.is_file()) == [
        "__pycache__/x.pyc", "app.py"]
