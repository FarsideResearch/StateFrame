"""Durable, ordered parallel storage for checkpoint tree objects."""
from __future__ import annotations

import contextvars
import os
import random
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe.engine import kernel_checkpoint as checkpoint


@pytest.fixture(autouse=True)
def _isolated_checkpoint_state(monkeypatch):
    monkeypatch.setattr(checkpoint, "_SAVE", {})
    monkeypatch.setattr(checkpoint, "_STATE", {})
    monkeypatch.setattr(checkpoint, "_VERIFIED_OBJECTS", set())
    monkeypatch.setattr(checkpoint, "_DIGEST_MEMO", {})
    monkeypatch.setattr(checkpoint, "_SECRETS", [])
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", 0)
    monkeypatch.setattr(checkpoint, "_check_registered_secrets", lambda *_args, **_kwargs: None)


def _files(root: Path, count: int, *, duplicate: bool = False) -> Path:
    root.mkdir(parents=True)
    for index in range(count):
        data = b"first shared body" if duplicate and index in {0, count - 1} else f"body-{index:04d}".encode()
        path = root / f"file-{index:04d}"
        path.write_bytes(data)
        os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
    return root


def _capture(source: Path, target: Path, objects: Path):
    digests: dict[str, str] = {}
    checkpoint._SAVE.update(
        snapshot_objects=objects, bytes_new=[0], copy_modes={}, copy_mtimes={},
        copy_digests=(digests, {}), copy_prefix="repo")
    counts = checkpoint._copy_tree(
        source, target, None, set(), objects=objects, home=None, home_subtree=None, excluded=set())
    tree = {str(path.relative_to(target)): path.read_bytes() for path in target.rglob("*") if path.is_file()}
    receipt = (counts, checkpoint._SAVE["bytes_new"][0], digests,
               checkpoint._SAVE["copy_modes"], checkpoint._SAVE["copy_mtimes"])
    return tree, receipt


def _pool_threads() -> list[str]:
    return [thread.name for thread in threading.enumerate() if thread.name.startswith("ThreadPoolExecutor")]


def _save_fixture(monkeypatch):
    frame = SimpleNamespace(f_code=(lambda: None).__code__, f_locals={})

    def collect(_frame):
        payload = {"modules": {}, "classes": {}, "patched": {}, "participants": {},
                   "context": [], "random": random.getstate(), "frame": {"value": "saved"}}
        context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
        return payload, context

    monkeypatch.setattr(checkpoint, "collect", collect)
    monkeypatch.setattr(checkpoint, "_workspace_objects_on", lambda: False)
    monkeypatch.setattr(checkpoint, "_verify_stateframe_home_identity", lambda _run: "test-identity")
    monkeypatch.setattr(checkpoint, "_snapshot_home_exclusions", lambda _run: (None, None, set()))
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"checkpoint-test-key-0123456789abcdef"])
    monkeypatch.setattr(checkpoint, "_atomic_checkpoints", lambda: False)
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _clocks: None)
    monkeypatch.setattr(checkpoint.contextvars, "copy_context", contextvars.Context)
    return frame


def test_save_with_existing_objects_scans_tree_once(tmp_path, monkeypatch):
    run = tmp_path / "run"
    source = _files(run / "repo", 9)
    frame = _save_fixture(monkeypatch)
    checkpoint.save(frame, 1, run)
    previous = checkpoint.latest(run)
    assert previous is not None
    previous_repo = previous / "run" / "repo"
    object_store = run / "checkpoints" / checkpoint.OBJECT_DIR
    previous_objects = {path.name for path in object_store.iterdir()}
    assert len(previous_objects) == 9
    counts = {"scandir": 0, "lstat": 0, "digest": 0}
    real_scandir, real_lstat = os.scandir, os.lstat
    real_digest = checkpoint._live_digest

    def scandir(path="."):
        if not isinstance(path, int) and os.fspath(path) == os.fspath(source):
            counts["scandir"] += 1
        return real_scandir(path)

    def lstat(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent == previous_repo and candidate.name.startswith("file-"):
            counts["lstat"] += 1
        return real_lstat(path, *args, **kwargs)

    def live_digest(path, info):
        if Path(path).parent == source:
            counts["digest"] += 1
        return real_digest(path, info)

    monkeypatch.setattr(checkpoint.os, "scandir", scandir)
    monkeypatch.setattr(checkpoint.os, "lstat", lstat)
    monkeypatch.setattr(checkpoint, "_live_digest", live_digest)
    checkpoint.save(frame, 2, run)
    assert counts == {"scandir": 1, "lstat": 9, "digest": 9}
    assert {path.name for path in object_store.iterdir()} == previous_objects


def test_parallel_fsync_precedes_replace_and_all_lifecycle_hooks(tmp_path, monkeypatch):
    source = _files(tmp_path / "source", 12)
    temp_for_fd: dict[int, str] = {}
    temp_names: list[str] = []
    temp_name_set: set[str] = set()
    events: list[tuple] = []
    lifecycle: list[str] = []
    real_mkstemp, real_fsync = checkpoint.tempfile.mkstemp, checkpoint.os.fsync
    real_replace, real_link = checkpoint.os.replace, checkpoint.os.link

    def mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        temp_for_fd[fd] = name
        temp_names.append(name)
        temp_name_set.add(name)
        return fd, name

    def fsync(fd):
        if fd in temp_for_fd:
            events.append(("fsync", temp_for_fd[fd]))
        return real_fsync(fd)

    def replace(src, dst, *args, **kwargs):
        if str(src) in temp_name_set:
            events.append(("replace", str(src), str(dst)))
        if Path(dst).name == "LATEST":
            assert not _pool_threads()
            lifecycle.append("LATEST")
        return real_replace(src, dst, *args, **kwargs)

    def link(src, dst, *args, **kwargs):
        if not any(event[0] == "link" for event in events):
            fsyncs = {event[1] for event in events if event[0] == "fsync"}
            replaces = {event[1] for event in events if event[0] == "replace"}
            assert len(fsyncs) == len(replaces) == 12
            for index, event in enumerate(events):
                if event[0] == "replace":
                    assert events.index(("fsync", event[1])) < index
            assert not _pool_threads()
        events.append(("link", str(src), str(dst)))
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(checkpoint.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(checkpoint.os, "fsync", fsync)
    monkeypatch.setattr(checkpoint.os, "replace", replace)
    monkeypatch.setattr(checkpoint.os, "link", link)
    _, temp_receipt = _capture(source, tmp_path / "snapshot" / "run" / "repo", tmp_path / "objects")
    assert len(temp_names) == len(set(temp_names)) == 12
    assert temp_receipt[0]["copied"] == 12

    def marker(*_args, **_kwargs):
        assert not _pool_threads()
        lifecycle.append("marker")

    root, temp, final = tmp_path / "publish", tmp_path / "publish" / "pending", tmp_path / "publish" / "saved"
    root.mkdir()
    temp.mkdir()
    (temp / checkpoint.ARTIFACT_TRANSACTION_FILE).touch()
    monkeypatch.setattr(checkpoint, "_atomic_checkpoints", lambda: False)
    monkeypatch.setattr(checkpoint, "_checkpoints", lambda _root: (final,))
    monkeypatch.setattr(checkpoint, "_keep", lambda: 1)
    monkeypatch.setattr(checkpoint, "_fsync_dir", lambda *_args: None)
    monkeypatch.setattr(checkpoint, "_fsync_file", lambda *_args: None)
    monkeypatch.setattr(checkpoint, "_prune_chunks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checkpoint, "_prune_pool", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(checkpoint, "_prune_objects", lambda *_args, **_kwargs: None)
    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.name == checkpoint.ARTIFACT_TRANSACTION_FILE:
            marker()
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    checkpoint._publish_checkpoint(root, temp, final, tmp_path)
    assert lifecycle == ["LATEST", "marker"]

    monkeypatch.setattr(checkpoint, "_writer_journal_start", lambda _run: (0, "journal", False))
    fork_threads = []

    def fake_fork():
        fork_threads.extend(thread.name for thread in threading.enumerate())
        assert not _pool_threads()
        lifecycle.append("fork")
        return os.getpid() + 10_000

    monkeypatch.setattr(checkpoint.os, "fork", fake_fork)
    checkpoint._fork_writer(tmp_path, 1, final, lambda: {}, final)
    assert lifecycle[-1] == "fork"
    assert not any(name.startswith("ThreadPoolExecutor") for name in fork_threads)


def test_parallel_and_forced_serial_tree_receipts_match_and_restore(tmp_path, monkeypatch):
    parallel_run, serial_run = tmp_path / "parallel-run", tmp_path / "serial-run"
    _files(parallel_run / "repo", 12, duplicate=True)
    _files(serial_run / "repo", 12, duplicate=True)
    frame = _save_fixture(monkeypatch)
    parallel_checkpoint = checkpoint.save(frame, 1, parallel_run)
    parallel_meta = checkpoint.validate(parallel_checkpoint)
    real_batch = checkpoint._store_object_batch
    monkeypatch.setattr(checkpoint, "_store_object_batch",
                        lambda batch, objects: real_batch(batch, objects, minimum=1000))
    checkpoint._STATE.clear()
    serial_checkpoint = checkpoint.save(frame, 1, serial_run)
    serial_meta = checkpoint.validate(serial_checkpoint)
    fields = ("files", "bytes_new", "file_sha256", "file_modes", "file_mtime_ns", "link_fallback")
    assert {name: parallel_meta[name] for name in fields} == {name: serial_meta[name] for name in fields}
    checkpoint.restore(frame, parallel_checkpoint, parallel_run)
    assert frame.f_locals["value"] == "saved"
    assert {p.name: p.read_bytes() for p in (parallel_run / "repo").iterdir()} == {
        p.name: p.read_bytes() for p in (serial_run / "repo").iterdir()}


def test_duplicate_digests_store_one_object(tmp_path):
    source = _files(tmp_path / "source", 9, duplicate=True)
    _capture(source, tmp_path / "snapshot" / "repo", tmp_path / "objects")
    assert len(list((tmp_path / "objects").iterdir())) == 8


def test_fsync_failure_matches_serial_class_cleans_temps_and_preserves_latest(tmp_path, monkeypatch):
    class InjectedSyncError(OSError):
        pass

    monkeypatch.setattr(checkpoint.os, "fsync", lambda _fd: (_ for _ in ()).throw(InjectedSyncError("sync failed")))
    serial_source = _files(tmp_path / "serial-source", 7)
    with pytest.raises(InjectedSyncError) as serial_error:
        _capture(serial_source, tmp_path / "serial-snapshot" / "repo", tmp_path / "serial-objects")
    run = tmp_path / "parallel-run"
    _files(run / "repo", 8)
    (run / "checkpoints").mkdir()
    latest = run / "checkpoints" / "LATEST"
    latest.write_text("previous\n")
    frame = _save_fixture(monkeypatch)
    monkeypatch.setattr(checkpoint, "latest", lambda _run: None)
    with pytest.raises(InjectedSyncError) as parallel_error:
        checkpoint.save(frame, 1, run)
    assert type(parallel_error.value) is type(serial_error.value)
    assert not list((tmp_path / "serial-objects").glob("*.tmp"))
    assert not list((run / "checkpoints" / checkpoint.OBJECT_DIR).glob("*.tmp"))
    assert latest.read_text() == "previous\n"


def test_source_changed_after_scan_is_refused_without_temp_files(tmp_path, monkeypatch):
    source = _files(tmp_path / "source", 8)
    changed = source / "file-0000"
    real_digest = checkpoint._live_digest
    mutated = False

    def digest(path, info):
        nonlocal mutated
        result = real_digest(path, info)
        if Path(path) == changed and not mutated:
            changed.write_bytes(b"mutate-0000")
            mutated = True
        return result

    monkeypatch.setattr(checkpoint, "_live_digest", digest)
    objects = tmp_path / "objects"
    with pytest.raises(checkpoint.CheckpointError, match="changed during checkpoint capture"):
        _capture(source, tmp_path / "snapshot" / "repo", objects)
    assert not list(objects.glob("*.tmp"))


def test_memory_error_in_worker_propagates_after_pool_shutdown(tmp_path, monkeypatch):
    source = _files(tmp_path / "source", 8)
    real_store = checkpoint._store_object

    def store(path, info, digest, objects):
        if Path(path).name == "file-0000":
            raise MemoryError("injected allocation failure")
        return real_store(path, info, digest, objects)

    monkeypatch.setattr(checkpoint, "_store_object", store)
    with pytest.raises(MemoryError, match="injected allocation failure"):
        _capture(source, tmp_path / "snapshot" / "repo", tmp_path / "objects")
    assert not _pool_threads()


def test_marked_stop_prevents_queued_objects_from_starting(tmp_path, monkeypatch):
    class MarkedStop(RuntimeError):
        boundary_recovery_stop = True

    candidates = [(str(tmp_path / f"file-{i:03d}"), None, str(i), f"object-{i}") for i in range(64)]
    stop_seen = threading.Event()
    calls: list[str] = []

    class ObservedEvent:
        def __init__(self):
            self.value = threading.Event()

        def is_set(self):
            return self.value.is_set()

        def set(self):
            self.value.set()
            stop_seen.set()

    monkeypatch.setattr(checkpoint, "threading", SimpleNamespace(Event=ObservedEvent))

    def store(path, *_args):
        name = Path(path).name
        calls.append(name)
        if name == "file-000":
            raise MarkedStop("stop now")
        stop_seen.wait()
        return Path(path), 0

    monkeypatch.setattr(checkpoint, "_store_object", store)
    with pytest.raises(MarkedStop, match="stop now"):
        checkpoint._store_object_batch(candidates, tmp_path / "objects")
    assert calls and set(calls) <= {f"file-{i:03d}" for i in range(16)}
    assert not _pool_threads()


def test_main_thread_interrupt_stops_queued_objects_and_cleans_temps(tmp_path, monkeypatch):
    source = _files(tmp_path / "source", 32)
    objects, calls = tmp_path / "objects", []
    release, started = threading.Event(), threading.Event()
    real_store, real_fsync = checkpoint._store_object, checkpoint.os.fsync
    real_submit, submitted = checkpoint.ThreadPoolExecutor.submit, 0

    class StopEvent:
        def is_set(self): return release.is_set()
        def set(self): release.set()

    def fsync(fd):
        if threading.current_thread().name.startswith("ThreadPoolExecutor"):
            started.set()
            release.wait(2)
        return real_fsync(fd)

    def store(path, *args):
        calls.append(path)
        return real_store(path, *args)

    def submit(pool, *args, **kwargs):
        nonlocal submitted
        submitted += 1
        if submitted == 24:
            assert started.wait(5)
            raise KeyboardInterrupt
        return real_submit(pool, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "threading", SimpleNamespace(Event=StopEvent))
    monkeypatch.setattr(checkpoint.os, "fsync", fsync)
    monkeypatch.setattr(checkpoint, "_store_object", store)
    monkeypatch.setattr(checkpoint.ThreadPoolExecutor, "submit", submit)
    batch = [(str(path), info := path.stat(), checkpoint._live_digest(str(path), info)[1], str(index))
             for index, path in enumerate(source.iterdir())]
    with pytest.raises(KeyboardInterrupt):
        checkpoint._store_object_batch(batch, objects)
    assert len(calls) <= 16
    assert not list(objects.glob("*.tmp"))
    assert not _pool_threads()
