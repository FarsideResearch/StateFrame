"""Work budgets for fenced workspace snapshots and the shared object store."""
from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from threading import Event
from time import perf_counter, time_ns
from types import SimpleNamespace

import pytest


def _tree_codec(monkeypatch, objects):
    """Use the separately owned codec when it is available in the stitched tree."""
    try:
        import kernel_workspace_tree_io  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "kernel_workspace_tree_io":
            raise
    else:
        return
    from kernel_workspace_tree import _DOMAIN, _field, _uint

    def write_tree(lease, tree, *, previous=None):
        digest = tree.digest.hex()
        if previous != digest:
            page = tree._page.digest if tree._page is not None else b""
            data = _DOMAIN + _field(b"dir") + _field(_uint(tree.mode)) + _field(page)
            lease.publish_many([(objects.ObjectRef(digest, len(data)), lambda: io.BytesIO(data))])
        return digest

    monkeypatch.setitem(sys.modules, "kernel_workspace_tree_io", SimpleNamespace(write_tree=write_tree))


def test_repeated_pins_and_sparse_capture(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    home = tmp_path / "objects"
    home.mkdir(mode=0o700)
    payload = b"a" * 1024
    for index in range(5_000):
        (root / f"file-{index:05d}").write_bytes(payload)
    try:
        import kernel_workspace_index as index_module
        import kernel_workspace_objects as objects
        from kernel_workspace_state import WorkspaceFence, WorkspaceState
    except ModuleNotFoundError as error:
        if error.name not in ("kernel_workspace_index", "kernel_workspace_objects",
                              "kernel_workspace_state"):
            raise
        from kernel_verdict_identity import workspace_tree_hash

        workspace_tree_hash(root)
        original_read = Path.read_bytes
        raw_reads = 0

        def counted(path):
            nonlocal raw_reads
            raw_reads += 1
            return original_read(path)

        monkeypatch.setattr(Path, "read_bytes", counted)
        started = perf_counter()
        workspace_tree_hash(root)
        print(f"pinned release clean={perf_counter() - started:.6f}s source_opens={raw_reads}")
        assert raw_reads == 0, "clean capture reopened source files"
        return
    _tree_codec(monkeypatch, objects)
    # Simulate a long-settled starting worktree without a timed sleep.
    monkeypatch.setattr(index_module, "time_ns", lambda: time_ns() + 5_000_000_000)
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"), trust_stat=True)
    store = objects.open_workspace_objects(home, "a" * 64, enabled=True)
    assert store is not None
    counts = Counter()

    def spy(owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    spy(os, "scandir")
    spy(objects, "_read_checked")
    spy(os, "link")
    spy(os, "read")
    spy(os, "fsync")

    def capture():
        with state.pin("capture") as snapshot, store.lease() as lease:
            if hasattr(snapshot, "persist"):
                return snapshot.persist(lease)
            # The prerequisite tree has an identity-only index. Capturing a
            # reusable byte snapshot still needs a full live source verification.
            from kernel_workspace_tree import children

            entries = [
                (objects.ObjectRef(entry.sha256.hex(), entry.size),
                 lambda name=name: (root / os.fsdecode(name)).open("rb"))
                for name, entry in children(snapshot.tree)
            ]
            current = state._index.refresh(opaque_write=True)
            lease.publish_many(entries)
            return current.digest.hex()

    started = perf_counter()
    original = capture()
    print(f"cold={perf_counter() - started:.6f}s {dict(counts)}")
    clean_times = []
    for _ in range(200):
        counts.clear()
        started = perf_counter()
        assert capture() == original
        clean_times.append(perf_counter() - started)
        if counts or state.work.content_reads:
            print(f"clean={clean_times[-1]:.6f}s {dict(counts)} reads={state.work.content_reads}")
        assert counts == {}, f"clean capture repeated filesystem work: {dict(counts)}"
        assert state.work.content_reads == 0
    print(f"clean_200={sum(clean_times):.6f}s max={max(clean_times):.6f}s")
    assert max(clean_times) < 0.5

    counts.clear()
    path = root / "file-02500"
    before = path.stat()
    with state.begin_write((path.name,)):
        path.write_bytes(b"b" * 1024)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    started = perf_counter()
    changed = capture()
    print(f"sparse={perf_counter() - started:.6f}s {dict(counts)}")
    assert changed != original
    assert state.work.content_reads == 1
    assert state.work.rehashed_bytes == 1024
    assert counts["scandir"] == 0
    assert counts["link"] <= 48
    blob = objects.ObjectRef(hashlib.sha256(b"b" * 1024).hexdigest(), 1024)
    with store.lease() as lease:
        assert lease.read(blob) == b"b" * 1024

    with state.begin_write():
        path.write_bytes(b"c" * 1024)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert capture() not in (original, changed)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    refs = pytest.importorskip("kernel_workspace_refs")
    import kernel_workspace_objects as objects

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    namespace = "a" * 64
    store = refs.bind_run_store(run, namespace)
    return objects, refs, store, run


def test_interrupted_stream_retry_and_collection_hold_published_root(registry, monkeypatch):
    objects, refs, store, _ = registry
    data = b"x" * (objects.MAX_OBJECT_BYTES + 1)
    oid = hashlib.sha256(data).hexdigest()
    key = refs.RootKey("run", "checkpoint", "old")
    target = refs.Ref("blob", oid)
    original_link = os.link
    fsync = os.fsync
    linked = False

    def link(source, destination, **kwargs):
        nonlocal linked
        original_link(source, destination, **kwargs)
        if destination == oid:
            linked = True

    def interrupted(fd):
        if linked:
            raise OSError("interrupted before object directory sync")
        fsync(fd)

    with store.lease() as lease, monkeypatch.context() as patch:
        patch.setattr(os, "link", link)
        patch.setattr(os, "fsync", interrupted)
        with pytest.raises(OSError, match="interrupted before"):
            lease.publish_stream(io.BytesIO(data), size=len(data), max_bytes=len(data))
        assert refs.read_root(lease, key) is None
    pool = store._path / "objects"
    assert (pool / (".stage-" + oid)).is_file()
    entered = Event()

    def collect():
        entered.set()
        return store.collect(lambda _lease: ())

    with ThreadPoolExecutor(max_workers=1) as executor:
        with store.lease() as lease:
            future = executor.submit(collect)
            assert entered.wait(5)
            assert not future.done()
            stored = lease.publish_stream(io.BytesIO(data), size=len(data), max_bytes=len(data))
            assert stored == objects.LargeObjectRef(oid, len(data))
            assert lease.resolve(oid, max_bytes=len(data)) == stored
            with pytest.raises(objects.ObjectIntegrityError, match="bounded read"):
                lease.read(stored)
            with pytest.raises(objects.ObjectIntegrityError, match="permitted size"):
                lease.resolve(oid, max_bytes=len(data) - 1)
            refs.commit_root(lease, key, target, expected=None)
        assert future.result(timeout=10) == 0
    assert {entry.name for entry in pool.iterdir()} == {oid}
    with store.lease() as lease:
        output = io.BytesIO()
        lease.copy_verified(stored, output, max_bytes=len(data))
        assert output.getvalue() == data
        refs.retire_root(lease, key, expected=target)
    assert store.collect(lambda _lease: ()) == 1


@pytest.mark.parametrize(("payload", "size"), [(b"short", 6), (b"long", 3)])
def test_stream_bounds_leave_no_address_or_stage(registry, payload, size):
    objects, _, store, _ = registry
    with store.lease() as lease, pytest.raises(objects.ObjectIntegrityError):
        lease.publish_stream(io.BytesIO(payload), size=size, max_bytes=16)
    assert list((store._path / "objects").iterdir()) == []


def test_binding_and_root_conflict_or_corruption_preserve_retention(registry, tmp_path, monkeypatch):
    objects, refs, store, run = registry
    assert not refs.workspace_objects_enabled({})
    assert refs.workspace_objects_enabled({"STATEFRAME_WORKSPACE_OBJECTS": " On "})
    with pytest.raises(ValueError):
        refs.workspace_objects_enabled({"STATEFRAME_WORKSPACE_OBJECTS": "yes"})
    archive = tmp_path / "archived"
    run.rename(archive)
    assert refs.open_run_store(archive) == store
    assert refs.workspace_run_id(archive) == "run"
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    descriptor = replacement / refs.RUN_BINDING
    descriptor.write_bytes((archive / refs.RUN_BINDING).read_bytes())
    descriptor.chmod(0o600)
    with pytest.raises(objects.ObjectIntegrityError, match="authentication"):
        refs.open_run_store(replacement)

    key = refs.RootKey("run", "candidate", "candidate")
    with store.lease() as lease:
        old = lease.publish_stream(io.BytesIO(b"old"), size=3, max_bytes=3)
        new = lease.publish_stream(io.BytesIO(b"new"), size=3, max_bytes=3)
        old_ref, new_ref = refs.Ref("blob", old.digest), refs.Ref("blob", new.digest)
        refs.commit_root(lease, key, old_ref, expected=None)
        refs.commit_root(lease, key, old_ref, expected=None)
        with pytest.raises(refs.RootConflictError):
            refs.commit_root(lease, key, new_ref, expected=None)
        with monkeypatch.context() as patch:
            def fail_rename(*args, **kwargs):
                raise OSError("root rename fault")

            patch.setattr(os, "replace", fail_rename)
            with pytest.raises(OSError, match="root rename fault"):
                refs.commit_root(lease, key, new_ref, expected=old_ref)
        assert refs.read_root(lease, key) == old_ref
    record = store._path / "refs" / key.run_id / key.kind / key.key
    record.write_bytes(record.read_bytes().replace(old.digest.encode(), new.digest.encode()))
    with pytest.raises(objects.ObjectIntegrityError, match="authentication"):
        store.collect(lambda _lease: ())
    assert {entry.name for entry in (store._path / "objects").iterdir()} == {old.digest, new.digest}


def test_collection_uses_typed_edges_and_prunes_abandoned_stages(registry, monkeypatch):
    _, refs, store, _ = registry
    from kernel_workspace_tree import _DOMAIN, _field, _uint

    def encoded(tag, *fields):
        return _DOMAIN + b"".join(map(_field, (tag, *fields)))

    def digest(data):
        return hashlib.sha256(data).digest()

    payload = b"tree data"
    file = encoded(b"file", _uint(0o600), _uint(len(payload)), digest(payload), b"\0")
    page = encoded(b"bucket", _field(b"file") + digest(file))
    root = encoded(b"dir", _uint(0o700), digest(page))
    by_kind = (
        refs.Ref("tree", digest(root).hex()),
        refs.Ref("node", digest(page).hex()),
        refs.Ref("node", digest(file).hex()),
        refs.Ref("blob", digest(payload).hex()),
    )
    try:
        import kernel_workspace_tree_io  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name != "kernel_workspace_tree_io":
            raise
        edges = dict(pairwise(by_kind))

        def child_refs(lease, ref):
            stored = lease.resolve(ref.oid, max_bytes=256)
            lease.read(stored)
            yield edges[ref]

        monkeypatch.setitem(sys.modules, "kernel_workspace_tree_io", SimpleNamespace(child_refs=child_refs))

    tree_key = refs.RootKey("run", "checkpoint", "tree")
    raw_key = refs.RootKey("run", "checkpoint", "raw")
    with store.lease() as lease:
        for data in (payload, file, page, root, b"orphan"):
            lease.publish_stream(io.BytesIO(data), size=len(data), max_bytes=len(data))
        refs.commit_root(lease, raw_key, refs.Ref("blob", by_kind[0].oid), expected=None)
        refs.commit_root(lease, tree_key, by_kind[0], expected=None)
    index = store._path / "index"
    index.mkdir(mode=0o700)
    stage = index / (".index-" + "b" * 32)
    stage.write_bytes(b"interrupted cache rebuild")
    root_stage = store._path / "refs" / "run" / "checkpoint" / (".root-" + "c" * 32)
    root_stage.write_bytes(b"interrupted root write")
    pool = store._path / "objects"
    published_stage = pool / (".stage-" + by_kind[0].oid)
    os.link(pool / by_kind[0].oid, published_stage)
    assert store.collect(lambda _lease: ()) == 1
    assert not stage.exists() and not root_stage.exists() and not published_stage.exists()
    with store.lease() as lease:
        refs.retire_root(lease, tree_key, expected=by_kind[0])
    assert store.collect(lambda _lease: ()) == 3
    assert {path.name for path in (store._path / "objects").iterdir()} == {by_kind[0].oid}


def test_disposable_rows_follow_sparse_deletions_and_modes(registry, tmp_path, monkeypatch):
    objects, _, store, _ = registry
    _tree_codec(monkeypatch, objects)
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "before").write_bytes(b"content")
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))

    def persist():
        with state.pin("index") as snapshot, store.lease() as lease:
            return snapshot.persist(lease)

    original = persist()
    with state.begin_write(("before", "after")):
        (root / "before").rename(root / "after")
        (root / "after").chmod(0o700)
    updated = persist()
    assert original != updated
    database_path, = (store._path / "index").glob("*.sqlite")
    with sqlite3.connect(database_path) as database:
        assert database.execute("SELECT tree FROM head").fetchone() == (updated,)
        entries = database.execute("SELECT path, mode, blob FROM entries ORDER BY path").fetchall()
    assert [entry[0] for entry in entries] == [b"", b"after"]
    assert entries[1][1] & 0o777 == 0o700
    assert entries[1][2] == hashlib.sha256(b"content").digest()
    database_path.write_bytes(b"corrupt disposable index")
    with state.begin_write(("added",)):
        (root / "added").write_bytes(b"extra")
    updated = persist()
    with sqlite3.connect(database_path) as database:
        assert database.execute("SELECT tree FROM head").fetchone() == (updated,)
        assert database.execute("SELECT path FROM entries ORDER BY path").fetchall() == [
            (b"",), (b"added",), (b"after",),
        ]


def test_cas_hit_never_bypasses_rotated_decoded_secret_policy(registry, tmp_path, monkeypatch):
    objects, _, store, _ = registry
    _tree_codec(monkeypatch, objects)
    import kernel_workspace_state as state_module

    root = tmp_path / "workspace"
    root.mkdir()
    policy = ()
    monkeypatch.setattr(state_module, "_secret_policy", lambda: policy)
    token = b"private-test-value"
    encoded = b"".join(f"\\u{byte:04x}".encode() for byte in token)
    data = b"x" * ((256 << 10) - 4) + encoded
    (root / "binary").write_bytes(data)
    state = state_module.WorkspaceState(state_module.WorkspaceFence.create(root, tmp_path / "fence"))
    with state.pin("before") as snapshot, store.lease() as lease:
        snapshot.persist(lease)
    policy = (token,)
    with (
        state.pin("after") as snapshot,
        store.lease() as lease,
        pytest.raises(state_module.WorkspaceUnavailable, match="retention policy"),
    ):
        snapshot.persist(lease)
    assert state.work.content_reads == 1


def test_racy_tuple_is_rechecked_even_with_empty_footprint(registry, tmp_path, monkeypatch):
    objects, _, store, _ = registry
    _tree_codec(monkeypatch, objects)
    import kernel_workspace_index as index_module
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source"
    source.write_bytes(b"old")
    observed_at = source.stat().st_ctime_ns
    monkeypatch.setattr(index_module, "time_ns", lambda: observed_at)
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"), trust_stat=True)
    with state.pin("before") as snapshot, store.lease() as lease:
        old = snapshot.persist(lease)
    source.write_bytes(b"new")
    with state.pin("racy") as snapshot, store.lease() as lease:
        assert snapshot.persist(lease) != old
    assert state.work.content_reads == 1


def test_workspace_below_home_runs_is_admitted_but_home_link_is_refused(registry, monkeypatch):
    objects, _, store, run = registry
    _tree_codec(monkeypatch, objects)
    from kernel_workspace_state import (
        WorkspaceFence,
        WorkspaceState,
        WorkspaceUnavailable,
    )

    home = Path(os.environ["STATEFRAME_HOME"])
    workspace = home / "runs" / run.name / "work"
    workspace.mkdir(parents=True)
    (workspace / "data").write_bytes(b"permitted run data")
    state = WorkspaceState(WorkspaceFence.create(workspace, run / "fence"))
    with state.pin("work") as snapshot, store.lease() as lease:
        snapshot.persist(lease)
    with state.begin_write(("private",)):
        (workspace / "private").symlink_to(home / "identity.key")
    with (
        state.pin("private") as snapshot,
        store.lease() as lease,
        pytest.raises(WorkspaceUnavailable, match="retention policy"),
    ):
        snapshot.persist(lease)


def test_restore_metadata_updates_without_changing_semantic_root(registry, tmp_path, monkeypatch):
    objects, refs, store, _ = registry
    _tree_codec(monkeypatch, objects)
    import kernel_workspace_tree_io as codec
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "source"
    source.write_bytes(b"stable")
    state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
    rows_seen = []
    writer = getattr(codec, "write_restore_meta", None)

    def track(lease, rows, *, previous, removed=()):
        rows = list(rows)
        rows_seen.append((rows, previous))
        if writer is not None:
            return writer(lease, rows, previous=previous, removed=removed)
        raw = repr(rows).encode()
        stored = lease.publish_stream(io.BytesIO(raw), size=len(raw), max_bytes=len(raw))
        return refs.Ref("meta", stored.digest)

    monkeypatch.setattr(codec, "write_restore_meta", track, raising=False)
    with state.pin("first") as snapshot, store.lease() as lease:
        tree, meta = snapshot.persist_with_metadata(lease)
    before = source.stat()
    stamp = before.st_mtime_ns + 5_000_000_000
    with state.begin_write(("source",)):
        os.utime(source, ns=(before.st_atime_ns, stamp))
    with state.pin("touch") as snapshot, store.lease() as lease:
        current_tree, current_meta = snapshot.persist_with_metadata(lease)
    assert current_tree == tree and current_meta != meta
    assert rows_seen[1][1] == meta
    assert ((b"source",), stamp, None) in rows_seen[1][0]
    with state.pin("clean") as snapshot, store.lease() as lease:
        assert snapshot.persist_with_metadata(lease) == (tree, current_meta)
    assert len(rows_seen) == 2
