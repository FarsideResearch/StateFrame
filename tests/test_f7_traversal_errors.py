import errno
import os
from pathlib import Path

import pytest

from stateframe import runner, workcopy
from stateframe.engine import kernel_checkpoint
from stateframe.test_runners import discover_check


def _make_work(root: Path) -> workcopy.WorkCopy:
    repo = root / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    (tests / "test_app.py").write_text("def test_app():\n    assert False\n", encoding="utf-8")
    return workcopy.create(repo, root / "copy")


def _block_scandir(monkeypatch: pytest.MonkeyPatch, blocked: Path) -> None:
    original_scandir = os.scandir

    def scandir(path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> os.ScandirIterator[str]:
        if isinstance(path, int):
            info, target = os.fstat(path), blocked.stat()
            matches = (info.st_dev, info.st_ino) == (target.st_dev, target.st_ino)
        else:
            matches = Path(path).resolve() == blocked.resolve()
        if matches:
            raise PermissionError(errno.EACCES, "blocked by test", str(blocked))
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)


def test_create_refuses_to_skip_an_unreadable_test_tree(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    (tests / "test_app.py").write_text("def test_app(): pass\n", encoding="utf-8")
    _block_scandir(monkeypatch, tests)

    with pytest.raises(workcopy.WorkCopyError, match=str(tests)) as error:
        workcopy.create(repo, tmp_path / "copy")
    assert type(error.value).__name__ == "WorkCopyScanError"


def test_changed_files_refuses_to_report_deletions_after_a_scan_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = _make_work(tmp_path)
    blocked = work.copy / "tests"
    _block_scandir(monkeypatch, blocked)

    with pytest.raises(workcopy.WorkCopyError, match=str(blocked)) as error:
        workcopy.changed_files(work)
    assert type(error.value).__name__ == "WorkCopyScanError"


def test_candidate_digest_refuses_an_incomplete_tree(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    work = _make_work(tmp_path)
    blocked = work.copy / "tests"
    _block_scandir(monkeypatch, blocked)

    with pytest.raises(workcopy.WorkCopyError, match=str(blocked)) as error:
        runner.candidate_digest(work.copy)
    assert type(error.value).__name__ == "WorkCopyScanError"


def test_final_diff_is_not_published_after_a_scan_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = _make_work(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    blocked = work.copy / "tests"
    _block_scandir(monkeypatch, blocked)
    monkeypatch.setattr(kernel_checkpoint, "verify_stateframe_home_identity",
                        lambda *_args, **_kwargs: None)

    with pytest.raises(workcopy.WorkCopyError, match=str(blocked)) as error:
        runner._export_final_diff(work, run_dir, expected_identity=None)

    assert type(error.value).__name__ == "WorkCopyScanError"
    assert not (run_dir / "final.diff").exists()


def test_apply_in_place_refuses_to_write_after_a_scan_error(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work = _make_work(tmp_path)
    (work.copy / "app.py").write_text("value = 2\n", encoding="utf-8")
    original_app = work.original / "app.py"
    original_test = work.original / "tests" / "test_app.py"
    blocked = work.copy / "tests"
    _block_scandir(monkeypatch, blocked)

    with pytest.raises(workcopy.WorkCopyError, match=str(blocked)) as error:
        workcopy.apply_in_place(work)
    assert type(error.value).__name__ == "WorkCopyScanError"

    assert original_app.read_text(encoding="utf-8") == "value = 1\n"
    assert original_test.is_file()


def test_discover_check_refuses_to_skip_an_unreadable_test_tree(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    tests = repo / "tests"
    tests.mkdir(parents=True)
    (repo / "pyproject.toml").touch()
    (repo / "app.py").touch()
    (tests / "test_app.py").write_text("def test_app(): assert False\n", encoding="utf-8")
    _block_scandir(monkeypatch, tests)

    with pytest.raises(ValueError, match=str(tests)) as error:
        discover_check(repo)
    assert type(error.value).__name__ == "TestDiscoveryError"
