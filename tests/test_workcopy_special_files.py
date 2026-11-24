"""Special files in a user's repository are skipped without opening them as data."""
from __future__ import annotations

import errno
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import pytest

from _timeouts import scaled
from stateframe import workcopy
from stateframe.workcopy import WorkCopyError, WorkCopyScanError, apply_in_place, changed_files, create


ROOT = Path(__file__).resolve().parents[1]


def _create_worker(repo: str, destination: str) -> None:
    result = create(Path(repo), Path(destination), admit_snapshot=False)
    print(json.dumps([result.skipped, sorted(result.baseline)]))


def _no_files_worker(repo: str, destination: str) -> None:
    try:
        print(json.dumps(sorted(create(Path(repo), Path(destination), admit_snapshot=False).baseline)))
    except WorkCopyError as error:
        print(json.dumps(str(error)))


def _fifo_race_worker(repo: str, destination: str) -> None:
    repo_path = Path(repo)
    real_open = os.open
    real_fstat = os.fstat
    real_fdopen = os.fdopen
    opened_flags = []
    saw_fifo_descriptor = []
    read_special_descriptor = []

    class TrackedFile:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, *args, **kwargs):
            if not stat.S_ISREG(real_fstat(self.stream.fileno()).st_mode):
                read_special_descriptor.append(True)
            return self.stream.read(*args, **kwargs)

    def replace_regular_with_fifo(path, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(path) == "raced.py" and dir_fd is not None:
            os.unlink(repo_path / "raced.py")
            os.mkfifo(repo_path / "raced.py")
            opened_flags.append(flags)
        kwargs = {"dir_fd": dir_fd} if dir_fd is not None else {}
        return real_open(path, flags, mode, **kwargs)

    def record_fstat(fd):
        info = real_fstat(fd)
        if stat.S_ISFIFO(info.st_mode):
            saw_fifo_descriptor.append(True)
        return info

    def track_fdopen(fd, *args, **kwargs):
        return TrackedFile(real_fdopen(fd, *args, **kwargs))

    os.open = replace_regular_with_fifo
    os.fstat = record_fstat
    os.fdopen = track_fdopen
    try:
        result = create(repo_path, Path(destination), admit_snapshot=False)
    finally:
        os.open = real_open
        os.fstat = real_fstat
        os.fdopen = real_fdopen
    print(json.dumps([result.skipped, sorted(result.baseline), opened_flags,
                      bool(saw_fifo_descriptor), bool(read_special_descriptor)]))


def _symlink_race_worker(repo: str, destination: str, external: str) -> None:
    repo_path = Path(repo)
    real_open = os.open
    opened_flags = []
    moved = False

    def replace_regular_with_symlink(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal moved
        if os.fspath(path) == "race.py" and dir_fd is not None and not moved:
            os.replace(repo_path / "race.py", external)
            os.symlink(external, repo_path / "race.py")
            moved = True
            opened_flags.append(flags)
        kwargs = {"dir_fd": dir_fd} if dir_fd is not None else {}
        return real_open(path, flags, mode, **kwargs)

    os.open = replace_regular_with_symlink
    try:
        result = create(repo_path, Path(destination), admit_snapshot=False)
    finally:
        os.open = real_open
    print(json.dumps([result.skipped, sorted(result.baseline), opened_flags, moved]))


def _socket_race_worker(repo: str, destination: str) -> None:
    repo_path = Path(repo)
    real_open = os.open
    opened_flags = []
    listeners = []

    def replace_regular_with_socket(path, flags, mode=0o777, *, dir_fd=None):
        if os.fspath(path) == "race.py" and dir_fd is not None:
            os.unlink(repo_path / "race.py")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(os.fspath(repo_path / "race.py"))
            listeners.append(listener)
            opened_flags.append(flags)
        kwargs = {"dir_fd": dir_fd} if dir_fd is not None else {}
        return real_open(path, flags, mode, **kwargs)

    os.open = replace_regular_with_socket
    try:
        result = create(repo_path, Path(destination), admit_snapshot=False)
    finally:
        os.open = real_open
        for listener in listeners:
            listener.close()
    print(json.dumps([result.skipped, sorted(result.baseline), opened_flags]))


def _run_worker(target_name: str, *args: str, timeout: float = 60.0):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])[sys.argv[2]](*sys.argv[3:])"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    try:
        child = subprocess.run(
            [sys.executable, "-c", code, os.fspath(Path(__file__).resolve()), target_name, *args],
            env=environment, capture_output=True, text=True, timeout=scaled(timeout))
    except subprocess.TimeoutExpired:
        pytest.fail("the helper hung")
    assert child.returncode == 0, child.stderr
    return json.loads(child.stdout)


def _repo_with_file(tmp_path: Path, content: str = "value = 1\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "keep.py").write_text(content, encoding="utf-8")
    return repo


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
@pytest.mark.parametrize("relative", ["fifo.py", "nested/fifo.py"])
def test_create_skips_fifo_without_a_writer(tmp_path, relative):
    repo = _repo_with_file(tmp_path)
    fifo = repo / relative
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo)

    skipped, baseline = _run_worker("_create_worker", str(repo), str(tmp_path / "copy"))

    assert skipped == [relative]
    assert baseline == ["keep.py"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_repository_with_only_fifo_starts_empty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    os.mkfifo(repo / "only.py")

    assert _run_worker("_no_files_worker", str(repo), str(tmp_path / "copy")) == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
@pytest.mark.parametrize("relative", ["fifo.py", "nested/fifo.py"])
def test_create_skips_fifo_even_when_a_writer_is_waiting(tmp_path, relative):
    repo = _repo_with_file(tmp_path)
    fifo = repo / relative
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo)

    def writer():
        with fifo.open("wb", buffering=0) as stream:
            stream.write(b"must not become a source file")

    child = threading.Thread(target=writer, daemon=True)
    child.start()
    work = create(repo, tmp_path / "copy", admit_snapshot=False)
    if child.is_alive():
        # A correct scanner never opens the FIFO, so release the waiting writer after the scan.
        release = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
        child.join(timeout=3)
        os.close(release)
    else:
        child.join(timeout=3)

    assert not child.is_alive()
    assert work.skipped == [relative]
    assert set(work.baseline) == {"keep.py"}
    assert not (work.copy / relative).exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
@pytest.mark.parametrize("relative", ["fifo.py", "nested/fifo.py"])
def test_snapshot_admission_skips_fifo(tmp_path, relative):
    repo = _repo_with_file(tmp_path)
    fifo = repo / relative
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo)

    work = create(repo, tmp_path / "copy", admit_snapshot=True)

    assert work.skipped == [relative]
    assert set(work.baseline) == {"keep.py"}
    assert not (work.copy / relative).exists()


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets require POSIX")
@pytest.mark.parametrize("admit_snapshot", [False, True])
def test_create_skips_unix_socket(tmp_path, admit_snapshot):
    repo = Path(tempfile.mkdtemp(prefix="sf-sock-", dir="/tmp"))  # AF_UNIX paths are short
    (repo / "keep.py").write_text("value = 1\n", encoding="utf-8")
    address = repo / "local.py"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(os.fspath(address.resolve()))
            work = create(repo, tmp_path / "copy", admit_snapshot=admit_snapshot)
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    assert work.skipped == ["local.py"]
    assert set(work.baseline) == {"keep.py"}
    assert not (work.copy / "local.py").exists()


@pytest.mark.skipif(not hasattr(os, "mknod"), reason="device nodes require POSIX")
@pytest.mark.parametrize("admit_snapshot", [False, True])
def test_create_skips_character_device_when_permitted(tmp_path, admit_snapshot):
    repo = _repo_with_file(tmp_path)
    device = repo / "device.py"
    try:
        os.mknod(device, stat.S_IFCHR | 0o600, os.makedev(1, 3))
    except OSError as error:
        pytest.skip(f"device nodes are unavailable: {error.errno}")

    work = create(repo, tmp_path / "copy", admit_snapshot=admit_snapshot)

    assert work.skipped == ["device.py"]
    assert set(work.baseline) == {"keep.py"}


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_regular_entry_replaced_by_fifo_after_listing_is_not_opened_blocking(tmp_path):
    repo = _repo_with_file(tmp_path)
    (repo / "raced.py").write_text("value = 2\n", encoding="utf-8")

    skipped, baseline, flags, saw_fifo_descriptor, read_special = _run_worker(
        "_fifo_race_worker", str(repo), str(tmp_path / "copy"))

    assert skipped == ["raced.py"]
    assert baseline == ["keep.py"]
    assert len(flags) == 1 and flags[0] & os.O_NONBLOCK
    assert saw_fifo_descriptor
    assert not read_special


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="O_NOFOLLOW requires POSIX")
def test_symlink_installed_after_listing_is_opened_with_nofollow(tmp_path):
    repo = _repo_with_file(tmp_path)
    race = repo / "race.py"
    race.write_text("private = 'source'\n", encoding="utf-8")
    external = tmp_path / "external.py"

    skipped, baseline, flags, moved = _run_worker(
        "_symlink_race_worker", str(repo), str(tmp_path / "copy"), str(external))

    assert moved
    assert skipped == ["race.py"]
    assert baseline == ["keep.py"]
    assert len(flags) == 1 and flags[0] & os.O_NOFOLLOW
    assert external.read_text(encoding="utf-8") == "private = 'source'\n"


def test_file_removed_after_listing_keeps_typed_scan_error(tmp_path, monkeypatch):
    repo = _repo_with_file(tmp_path)
    (repo / "removed.py").write_text("value = 2\n", encoding="utf-8")
    real_stat = os.stat
    removed = []

    def remove_before_stat(path, *args, **kwargs):
        if not isinstance(path, int) and Path(os.fspath(path)).name == "removed.py" and not removed:
            (repo / "removed.py").unlink()
            removed.append(True)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(workcopy.os, "stat", remove_before_stat)
    with pytest.raises(WorkCopyScanError, match="cannot inspect repository file"):
        create(repo, tmp_path / "copy", admit_snapshot=False)

    assert removed


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="Unix sockets require POSIX")
def test_regular_entry_replaced_by_socket_is_skipped(tmp_path):
    repo = Path(tempfile.mkdtemp(prefix="sf-sock-", dir="/tmp"))  # AF_UNIX paths are short
    (repo / "keep.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "race.py").write_text("value = 2\n", encoding="utf-8")
    try:
        skipped, baseline, flags = _run_worker(
            "_socket_race_worker", str(repo.resolve()), str(tmp_path / "copy"))
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    assert skipped == ["race.py"]
    assert baseline == ["keep.py"]
    assert len(flags) == 1 and flags[0] & os.O_NONBLOCK


@pytest.mark.parametrize("admit_snapshot", [False, True])
@pytest.mark.parametrize("quota_errno", [errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)])
def test_enospc_and_edquot_have_the_same_error_and_leave_no_partial_copy(
        tmp_path, monkeypatch, admit_snapshot, quota_errno):
    repo = _repo_with_file(tmp_path)
    destination = tmp_path / "copy"
    real_open = Path.open
    injected = []

    class PartialQuotaWrite:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def flush(self):
            self.stream.flush()

        def seek(self, *args):
            return self.stream.seek(*args)

        def truncate(self, *args):
            return self.stream.truncate(*args)

        def write(self, value):
            self.stream.write(value[:1])
            raise OSError(quota_errno, "injected quota failure")

    def fail_after_partial_write(path, mode="r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if path.name == "keep.py" and any(flag in mode for flag in ("w", "x", "a")) and not injected:
            injected.append(True)
            return PartialQuotaWrite(stream)
        return stream

    monkeypatch.setattr(Path, "open", fail_after_partial_write)
    with pytest.raises(WorkCopyError) as caught:
        create(repo, destination, admit_snapshot=admit_snapshot)

    assert injected
    assert str(caught.value).startswith("repository copy needs ")
    assert "bytes are free (injected quota failure);" in str(caught.value)
    assert not destination.exists()
    assert not list(tmp_path.glob(".workspace-start-*"))
    assert not list(tmp_path.glob(".workcopy-*"))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_changed_files_and_in_place_ignore_special_entries_in_the_copy(tmp_path):
    repo = _repo_with_file(tmp_path)
    original_fifo = repo / "original-fifo.py"
    os.mkfifo(original_fifo)
    work = create(repo, tmp_path / "copy", admit_snapshot=False)
    os.mkfifo(work.copy / "model-fifo.py")
    (work.copy / "keep.py").write_text("value = 2\n", encoding="utf-8")

    assert changed_files(work) == {"keep.py": "modified"}
    assert apply_in_place(work) == ["keep.py"]
    assert (repo / "keep.py").read_text(encoding="utf-8") == "value = 2\n"
    assert stat.S_ISFIFO(original_fifo.lstat().st_mode)
    assert stat.S_ISFIFO((work.copy / "model-fifo.py").lstat().st_mode)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_resume_recopy_uses_regular_parent_snapshot_when_original_has_fifo(tmp_path):
    from stateframe.resume import fresh_workcopy

    repo = _repo_with_file(tmp_path)
    os.mkfifo(repo / "fifo.py")
    initial = create(repo, tmp_path / "initial", admit_snapshot=False)
    run = tmp_path / "run"
    run.mkdir()
    shutil.copytree(initial.copy, run / "parent")

    resumed = fresh_workcopy(run, repo)

    assert resumed.baseline == initial.baseline
    assert not (resumed.copy / "fifo.py").exists()
    assert stat.S_ISFIFO((repo / "fifo.py").lstat().st_mode)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFOs require POSIX")
def test_real_cli_run_applies_regular_change_and_leaves_repository_fifo_alone(
        tmp_path, monkeypatch, capsys):
    from stateframe import cli
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    os.mkfifo(repo / "special.py")
    run = tmp_path / "run"
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
    server = serve(script)
    try:
        result = cli.main([
            "run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run), "--test-python", sys.executable, "--no-run-env",
            "--no-project-hooks", "--in-place", "--quiet",
        ])
    finally:
        server.shutdown()
        server.server_close()
    capsys.readouterr()

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert result == 0
    assert summary["succeeded"]
    assert summary["applied_in_place"] == ["shop/pricing.py"]
    assert "(100 - percent)" in (repo / "shop/pricing.py").read_text(encoding="utf-8")
    assert stat.S_ISFIFO((repo / "special.py").lstat().st_mode)
    assert not (run / "parent" / "special.py").exists()
    script.assert_consumed()
