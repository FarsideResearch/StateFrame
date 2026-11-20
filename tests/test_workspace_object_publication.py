"""Publication, recovery and work bounds for private workspace objects."""

import errno
import fcntl
import hashlib
import io
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from threading import Event
from time import perf_counter

import pytest

MODULE = "kernel_workspace_objects"


def test_warm_publication_work_scales_with_distinct_digests(tmp_path, monkeypatch):
    workspace = tmp_path / "input"
    workspace.mkdir()
    storage = tmp_path / "store"
    storage.mkdir(mode=0o700)
    file_count = 5_000
    unique_count = 32
    paths = []
    payloads = [str(index).encode().ljust(1024, b"x") for index in range(unique_count)]
    digests = [hashlib.sha256(data).hexdigest() for data in payloads]
    for index in range(file_count):
        if index % 100 == 0:
            directory = workspace / str(index // 100)
            directory.mkdir()
        path = directory / str(index % 100)
        path.write_bytes(payloads[index % unique_count])
        paths.append(path)

    @contextmanager
    def legacy_publication():
        # The pinned release already avoids writing warm hits, but checks each
        # source and store object separately after discovering the manifest.
        from kernel_effects import ContentStore, Entry

        store = ContentStore(storage)
        manifest = {
            path.relative_to(workspace).as_posix():
                Entry("file", 0o644, digests[index % unique_count], 1024)
            for index, path in enumerate(paths)
        }
        yield lambda: store.capture(workspace, manifest)

    @contextmanager
    def snapshot_publication(objects):
        store = objects.open_workspace_objects(storage, "fixture", enabled=True)
        assert store is not None
        refs = [objects.ObjectRef(digest, 1024) for digest in digests]
        entries = [
            (refs[index % unique_count], lambda path=path: path.open("rb"))
            for index, path in enumerate(paths)
        ]
        with store.lease() as lease:
            yield lambda: lease.publish_many(entries)

    try:
        objects = import_module(MODULE)
    except ModuleNotFoundError as error:
        if error.name != MODULE:
            raise
        publication = legacy_publication()
    else:
        publication = snapshot_publication(objects)

    counts = {"source_opens": 0, "reads": 0, "bytes_read": 0, "syncs": 0, "staged": 0}
    original_io_open = io.open
    original_open = os.open
    original_read = os.read
    original_fsync = os.fsync

    def count_source(path):
        if isinstance(path, (str, Path)) and Path(path).is_relative_to(workspace):
            counts["source_opens"] += 1

    def count_io_open(path, *args, **kwargs):
        count_source(path)
        return original_io_open(path, *args, **kwargs)

    def count_open(path, flags, *args, **kwargs):
        count_source(path)
        if flags & os.O_CREAT:
            counts["staged"] += 1
        return original_open(path, flags, *args, **kwargs)

    def count_read(fd, size):
        block = original_read(fd, size)
        counts["reads"] += 1
        counts["bytes_read"] += len(block)
        return block

    def count_fsync(fd):
        counts["syncs"] += 1
        return original_fsync(fd)

    with publication as publish, monkeypatch.context() as patch:
        patch.setattr(io, "open", count_io_open)
        patch.setattr(os, "open", count_open)
        patch.setattr(os, "read", count_read)
        patch.setattr(os, "fsync", count_fsync)

        started = perf_counter()
        publish()
        cold = perf_counter() - started, counts.copy()
        counts.update(dict.fromkeys(counts, 0))
        started = perf_counter()
        publish()
        warm = perf_counter() - started, counts.copy()

    print(f"5,000 paths, 32 digests: cold={cold[0]:.6f}s {cold[1]}, warm={warm[0]:.6f}s {warm[1]}")
    assert counts == {
        "source_opens": 0, "reads": unique_count * 2, "bytes_read": unique_count * 1024,
        "syncs": 0, "staged": 0,
    }
    assert warm[0] < 0.55, f"warm publication took {warm[0]:.6f}s"


@pytest.fixture
def objects():
    return pytest.importorskip(MODULE)


@pytest.fixture
def private_store(objects, tmp_path):
    parent = tmp_path / "store"
    parent.mkdir(mode=0o700)
    store = objects.open_workspace_objects(parent, "run", enabled=True)
    return store, parent / "run" / "objects"


def _entry(objects, data):
    return objects.ObjectRef(hashlib.sha256(data).hexdigest(), len(data)), lambda: io.BytesIO(data)


def _unneeded_source():
    pytest.fail("a verified hit opened the source")


def test_default_off_has_no_filesystem_access(objects, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", lambda *_args, **_kwargs: pytest.fail("opened a directory while off"))
        patch.setattr(os, "mkdir", lambda *_args, **_kwargs: pytest.fail("created a directory while off"))
        assert objects.open_workspace_objects(tmp_path / "absent", "run") is None
        assert objects.open_workspace_objects(tmp_path / "absent", "run", enabled=False) is None
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("mask", [0o200, 0o400])
def test_registry_creation_restores_required_owner_bits(objects, tmp_path, monkeypatch, mask):
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    parent = tmp_path / "store"
    parent.mkdir(mode=0o700)
    previous_mask = os.umask(mask)
    try:
        store = objects.open_workspace_objects(
            parent, "run", enabled=os.environ["STATEFRAME_WORKSPACE_OBJECTS"] == "on",
        )
        assert store is not None
        from kernel_workspace_refs import RootKey, _commit_bytes, _root_directory, _root_lock

        key = RootKey("run-id", "checkpoint", "current")
        with store.lease() as lease:
            with _root_directory(lease, key, create=True) as directories:
                with _root_lock(directories[0], key):
                    _commit_bytes(directories[0], key.key, b"registry record")
        for directory in (store._path, store._path / "objects", store._path / "refs",
                          store._path / "refs" / "run-id",
                          store._path / "refs" / "run-id" / "checkpoint"):
            mode = stat.S_IMODE(directory.stat().st_mode)
            assert mode & 0o700 == 0o700
            assert mode & 0o077 == 0
        lock = store._path / "refs" / "run-id" / "checkpoint" / ".lock-current"
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600
        record = store._path / "refs" / "run-id" / "checkpoint" / "current"
        assert stat.S_IMODE(record.stat().st_mode) == 0o600
    finally:
        os.umask(previous_mask)


def test_namespace_isolation_and_no_follow(objects, private_store, tmp_path):
    store, pool = private_store
    entry = _entry(objects, b"isolated")
    with store.lease() as lease:
        lease.publish_many([entry])
    parent = pool.parent.parent
    other = objects.open_workspace_objects(parent, "other-policy", enabled=True)
    with other.lease() as lease, pytest.raises(FileNotFoundError):
        lease.read(entry[0])
    (parent / "alias").symlink_to(pool.parent, target_is_directory=True)
    (tmp_path / "alias").symlink_to(parent, target_is_directory=True)
    for location, namespace in [(parent, "alias"), (tmp_path / "alias", "run")]:
        with pytest.raises(OSError):
            objects.open_workspace_objects(location, namespace, enabled=True)
    with pytest.raises(ValueError, match="namespace"):
        objects.open_workspace_objects(parent, "../escape", enabled=True)


@pytest.mark.parametrize("kind", ["same-size", "short", "large", "symlink", "fifo", "writable"])
def test_corrupt_or_unsafe_existing_objects_fail_closed(objects, private_store, tmp_path, kind):
    store, pool = private_store
    ref, source = _entry(objects, b"correct")
    with store.lease() as lease:
        lease.publish_many([(ref, source)])
        path = pool / ref.digest
        if kind == "symlink":
            path.unlink()
            outside = tmp_path / "outside"
            outside.write_bytes(b"correct")
            path.symlink_to(outside)
        elif kind == "fifo":
            path.unlink()
            os.mkfifo(path, 0o400)
        else:
            path.chmod(0o600)
            if kind != "writable":
                path.write_bytes({"same-size": b"corrupt", "short": b"x", "large": b"x" * 8}[kind])
                path.chmod(0o400)
        before = path.lstat()
        with pytest.raises((objects.ObjectIntegrityError, OSError)):
            lease.read(ref)
        with pytest.raises((objects.ObjectIntegrityError, OSError)):
            lease.publish_many([(ref, _unneeded_source)])
        assert path.lstat().st_ino == before.st_ino
        assert list(pool.iterdir()) == [path]


@pytest.mark.parametrize("data", [b"wrong!!", b"x", b"exceeds length"])
def test_bad_source_never_becomes_addressable(objects, private_store, data):
    store, pool = private_store
    ref, good_source = _entry(objects, b"correct")
    with store.lease() as lease:
        with pytest.raises(objects.ObjectIntegrityError):
            lease.publish_many([(ref, lambda: io.BytesIO(data))])
        assert list(pool.iterdir()) == []
        lease.publish_many([(ref, good_source)])
        assert lease.read(ref) == b"correct"


def test_conflicting_references_and_empty_content(objects, private_store):
    store, pool = private_store
    empty = _entry(objects, b"")
    with pytest.raises(objects.ObjectIntegrityError):
        objects.ObjectRef("/" * 64, 0)
    with pytest.raises(objects.ObjectIntegrityError):
        objects.ObjectRef(empty[0].digest, objects.MAX_OBJECT_BYTES + 1)
    with store.lease() as lease:
        with pytest.raises(objects.ObjectIntegrityError, match="conflicting"):
            lease.publish_many([empty, (objects.ObjectRef(empty[0].digest, 1), _unneeded_source)])
        assert not (pool / empty[0].digest).exists()
        lease.publish_many([empty])
        assert lease.read(empty[0]) == b""


_CHILD = """
import hashlib
import io
import os
import sys
from pathlib import Path

from kernel_workspace_objects import ObjectRef, open_workspace_objects

parent = Path(sys.argv[1])
store = open_workspace_objects(parent, "run", enabled=True)
kind, ordinal = sys.argv[2], int(sys.argv[3])

def source(data):
    print("loaded", flush=True)
    return io.BytesIO(data)

def entry(data):
    return ObjectRef(hashlib.sha256(data).hexdigest(), len(data)), lambda: source(data)

def stop_after(operation):
    calls = 0
    def wrapped(*args, **kwargs):
        nonlocal calls
        operation(*args, **kwargs)
        calls += 1
        if calls == ordinal:
            os._exit(93)
    return wrapped

if kind != "normal":
    setattr(os, kind, stop_after(getattr(os, kind)))
with store.lease() as lease:
    entries = [entry(b"new A")]
    if kind != "normal":
        entries.append(entry(b"new B"))
    lease.publish_many(entries)
"""


def _run_child(pool, operation="normal", occurrence=0):
    engine = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
    return subprocess.run(
        [sys.executable, "-c", _CHILD, str(pool.parent.parent), operation, str(occurrence)],
        env={**os.environ, "PYTHONPATH": str(engine)},
        capture_output=True, text=True, timeout=90, check=False,
    )


def test_independent_process_writers_publish_one_inode(objects, private_store, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    store, pool = private_store
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(_run_child, [pool] * 4))
    assert [result.returncode for result in results] == [0] * 4, results
    assert sum(result.stdout.count("loaded") for result in results) == 1
    ref, _ = _entry(objects, b"new A")
    path = pool / ref.digest
    assert list(pool.iterdir()) == [path]
    before = path.stat()
    assert (before.st_nlink, stat.S_IMODE(before.st_mode)) == (1, 0o400)
    with store.lease() as lease:
        assert lease.read(ref) == b"new A"
        lease.publish_many([(ref, _unneeded_source)])
    assert path.stat().st_ino == before.st_ino


@pytest.mark.parametrize("operation, occurrence", [("fsync", i) for i in range(1, 5)]
                         + [("link", 1), ("link", 2)])
def test_process_exit_after_each_batch_sync_or_link(objects, private_store, operation, occurrence):
    store, pool = private_store
    old = _entry(objects, b"previous root")
    new = [_entry(objects, b"new A"), _entry(objects, b"new B")]
    with store.lease() as lease:
        lease.publish_many([old])
    child = _run_child(pool, operation, occurrence)
    assert child.returncode == 93, child.stderr

    with store.lease() as lease:
        assert lease.read(old[0]) == b"previous root"
        present = {}
        for ref, _ in new:
            path = pool / ref.digest
            if path.exists():
                assert lease.read(ref) in (b"new A", b"new B")
                present[ref.digest] = path.stat().st_ino
            else:
                with pytest.raises(FileNotFoundError):
                    lease.read(ref)
        lease.publish_many(new)
        assert [lease.read(ref) for ref, _ in new] == [b"new A", b"new B"]
    assert {path.name for path in pool.iterdir()} == {old[0].digest, *(ref.digest for ref, _ in new)}
    for digest, inode in present.items():
        assert (pool / digest).stat().st_ino == inode


@pytest.mark.parametrize("operation", ["open", "link", "file-sync", "dir-sync"])
def test_enospc_keeps_last_root_and_can_retry(objects, private_store, monkeypatch, operation):
    store, pool = private_store
    previous = _entry(objects, b"previous")
    new = _entry(objects, b"new")
    real_open, real_fsync = os.open, os.fsync

    def no_space_open(path, flags, *args, **kwargs):
        if flags & os.O_CREAT:
            raise OSError(errno.ENOSPC, "store full")
        return real_open(path, flags, *args, **kwargs)

    def no_space_link(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "store full")

    directory_syncs = 0

    def no_space_sync(fd):
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_syncs += 1
            if operation == "dir-sync" and directory_syncs == 2:
                raise OSError(errno.ENOSPC, "store full")
        elif operation == "file-sync":
            raise OSError(errno.ENOSPC, "store full")
        real_fsync(fd)

    with store.lease() as lease:
        lease.publish_many([previous])
        with monkeypatch.context() as patch:
            if operation == "open":
                patch.setattr(os, "open", no_space_open)
            elif operation == "link":
                patch.setattr(os, "link", no_space_link)
            else:
                patch.setattr(os, "fsync", no_space_sync)
            with pytest.raises(OSError) as exhausted:
                lease.publish_many([new])
            assert exhausted.value.errno == errno.ENOSPC
        assert lease.read(previous[0]) == b"previous"

    retried_syncs = []

    def count_sync(fd):
        retried_syncs.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_fsync(fd)

    with store.lease() as lease, monkeypatch.context() as patch:
        patch.setattr(os, "fsync", count_sync)
        retry = (new[0], _unneeded_source) if operation == "dir-sync" else new
        lease.publish_many([retry])
        assert lease.read(new[0]) == b"new"
    if operation == "dir-sync":
        assert retried_syncs == [True]
    assert {path.name for path in pool.iterdir()} == {previous[0].digest, new[0].digest}


def _observe_gc_blocked(monkeypatch):
    tried = Event()
    flock = fcntl.flock

    def check_blocked(fd, operation):
        if operation == fcntl.LOCK_EX:
            with pytest.raises(BlockingIOError):
                flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            tried.set()
        flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", check_blocked)
    return tried


def test_collection_waits_for_reader(objects, private_store, monkeypatch):
    store, _ = private_store
    entry = _entry(objects, b"reader")
    with ThreadPoolExecutor(max_workers=1) as workers:
        with store.lease() as reader:
            reader.publish_many([entry])
            tried = _observe_gc_blocked(monkeypatch)
            collected = workers.submit(store.collect, lambda _lease: ())
            assert tried.wait(15)
            assert not collected.done()
            assert reader.read(entry[0]) == b"reader"
        assert collected.result(timeout=15) == 1
    with pytest.raises(RuntimeError, match="closed"):
        reader.read(entry[0])


def test_collection_waits_for_publish_and_reads_roots_under_fence(objects, private_store, monkeypatch):
    store, pool = private_store
    parent = _entry(objects, b"new")
    orphan = _entry(objects, b"orphan")
    durable_root = pool.parent / "root"

    def reachable(reader):
        assert reader.read(parent[0]) == b"new"
        assert durable_root.read_text() == parent[0].digest
        return (parent[0],)

    with ThreadPoolExecutor(max_workers=1) as workers:
        with store.lease() as publisher:
            publisher.publish_many([parent, orphan])
            tried = _observe_gc_blocked(monkeypatch)
            collected = workers.submit(store.collect, reachable)
            assert tried.wait(15)
            assert not collected.done()
            with durable_root.open("w") as output:
                output.write(parent[0].digest)
                output.flush()
                os.fsync(output.fileno())
            namespace_fd = os.open(pool.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(namespace_fd)
            finally:
                os.close(namespace_fd)
        assert collected.result(timeout=15) == 1
    assert {path.name for path in pool.iterdir()} == {parent[0].digest}


def test_collection_with_incomplete_roots_leaves_all_objects(objects, private_store):
    store, pool = private_store
    present = _entry(objects, b"present")
    missing = _entry(objects, b"missing")
    with store.lease() as lease:
        lease.publish_many([present])
    with pytest.raises(FileNotFoundError):
        store.collect(lambda _lease: (missing[0],))
    assert list(pool.iterdir()) == [pool / present[0].digest]


def test_interrupted_collection_preserves_staging_debt(objects, private_store, monkeypatch):
    store, pool = private_store
    ref, _ = _entry(objects, b"new A")
    assert _run_child(pool, "link", 1).returncode == 93
    stage = pool / (".stage-" + ref.digest)
    assert stage.exists()
    unlink = os.unlink

    def interrupted(name, *args, **kwargs):
        if name == ref.digest:
            raise OSError(errno.EIO, "interrupted sweep")
        return unlink(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "unlink", interrupted)
        with pytest.raises(OSError, match="interrupted sweep"):
            store.collect(lambda _lease: ())
    assert stage.exists()
    assert store.collect(lambda _lease: ()) == 1
    assert list(pool.iterdir()) == []


def test_restore_is_private_checked_and_never_replaces(objects, private_store, tmp_path):
    store, pool = private_store
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    ref, source = _entry(objects, b"original")
    stamp = 1_700_000_000_123_456_789
    with store.lease() as lease:
        lease.publish_many([(ref, source)])
        lease.restore(ref, first / "file", mode=0o700, mtime_ns=stamp)
        lease.restore(ref, second / "file")
        target = first / "file"
        assert stat.S_IMODE(target.stat().st_mode) == 0o700
        assert target.stat().st_mtime_ns == stamp
        assert len({path.stat().st_ino for path in (target, second / "file", pool / ref.digest)}) == 3
        target.write_bytes(b"modified")
        with pytest.raises(FileExistsError):
            lease.restore(ref, target)
        assert target.read_bytes() == b"modified"
        assert (second / "file").read_bytes() == lease.read(ref) == b"original"
        assert target.stat().st_nlink == (second / "file").stat().st_nlink == 1

        alias = tmp_path / "alias"
        alias.symlink_to(first, target_is_directory=True)
        with pytest.raises(OSError):
            lease.restore(ref, alias / "escape")
        assert sorted(path.name for path in first.iterdir()) == ["file"]

        stored = pool / ref.digest
        stored.chmod(0o600)
        stored.write_bytes(b"tampered")
        stored.chmod(0o400)
        with pytest.raises(objects.ObjectIntegrityError):
            lease.restore(ref, first / "bad")
        assert list(first.iterdir()) == [target]


@pytest.mark.parametrize("ordinal", [1, 2])
def test_store_initialization_retries_after_either_sync(objects, tmp_path, monkeypatch, ordinal):
    parent = tmp_path / "store"
    parent.mkdir(mode=0o700)
    with monkeypatch.context() as patch:
        _raise_after(patch, "fsync", ordinal)
        with pytest.raises(OSError, match="injected interruption"):
            objects.open_workspace_objects(parent, "run", enabled=True)
    store = objects.open_workspace_objects(parent, "run", enabled=True)
    entry = _entry(objects, b"retry")
    with store.lease() as lease:
        lease.publish_many([entry])
        assert lease.read(entry[0]) == b"retry"


@pytest.mark.parametrize(
    "operation, ordinal, published", [("fsync", 1, False), ("link", 1, True), ("fsync", 2, True)],
)
def test_restore_interruption_leaves_only_absent_or_complete_bytes(
    objects, private_store, tmp_path, monkeypatch, operation, ordinal, published,
):
    store, _ = private_store
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    destination = staging / "file"
    entry = _entry(objects, b"durable")
    with store.lease() as lease:
        lease.publish_many([entry])
        with monkeypatch.context() as patch:
            _raise_after(patch, operation, ordinal)
            with pytest.raises(OSError, match="injected interruption"):
                lease.restore(entry[0], destination)
        assert list(staging.iterdir()) == ([destination] if published else [])
        if published:
            assert destination.read_bytes() == b"durable"
            assert destination.stat().st_nlink == 1
        assert lease.read(entry[0]) == b"durable"


def _raise_after(patch, operation, ordinal):
    original = getattr(os, operation)
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == ordinal:
            raise OSError(errno.EIO, "injected interruption")
        return result

    patch.setattr(os, operation, interrupted)
