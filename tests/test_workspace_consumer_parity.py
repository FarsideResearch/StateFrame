"""Work budgets and parity for pinned workspace consumers."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import kernel_green_lock as green_lock
import kernel_ship_tree_smoke as smoke


def test_clean_green_lock_workspace_budget(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(f"module_{index:04d}.py" for index in range(5000))
    body = b"VALUE = 1\n"
    for path in paths:
        (root / path).write_bytes(body)
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root))
    if hasattr(green_lock, "_workspace_consumers"):
        from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
        consumers = WorkspaceConsumers(state)
        bind_consumers(root, consumers)

    expected = "sha256:" + hashlib.sha256(body).hexdigest()
    assert green_lock._workspace_product_file_digests(runtime, paths) == dict.fromkeys(paths, expected)

    real_read = Path.read_bytes
    reads = 0

    def counted(path):
        nonlocal reads
        if path.is_relative_to(root):
            reads += 1
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    started = time.perf_counter()
    for _ in range(4):
        assert green_lock._workspace_product_file_digests(runtime, paths)[paths[0]] == expected
    elapsed = time.perf_counter() - started
    print(f"warm green lock: {elapsed:.6f}s, {reads} source reads")
    assert reads == 0
    if hasattr(green_lock, "_workspace_consumers"):
        assert state.work.directories_scanned == state.work.content_reads == 0
    assert elapsed < 2


def test_smoke_root_provider_work(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    package = root / "src"
    package.mkdir(parents=True)
    (package / "local_dep.py").write_text("VALUE = 1\n")
    paths = tuple(f"src/module_{index:04d}.py" for index in range(5000))
    for index, path in enumerate(paths):
        second = "import absent_dep\n" if index % 5 == 0 else ""
        (root / path).write_text("import local_dep\n" + second)
    monkeypatch.setenv("PYTHONPATH", "")
    if hasattr(smoke, "_workspace_consumers"):
        from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
        consumers = WorkspaceConsumers(state)
        bind_consumers(root, consumers)
    calls = 0
    real_provides = smoke._provides

    def counted(directory, name):
        nonlocal calls
        calls += 1
        return real_provides(directory, name)

    monkeypatch.setattr(smoke, "_provides", counted)
    started = time.perf_counter()
    roots = smoke.source_roots(root, paths)
    elapsed = time.perf_counter() - started
    print(f"cold smoke roots: {elapsed:.6f}s, {calls} provider probes")
    assert roots == (str(root), str(package))
    assert calls <= 8
    assert elapsed < 5
    calls = 0
    started = time.perf_counter()
    assert smoke.source_roots(root, paths) == roots
    elapsed = time.perf_counter() - started
    print(f"warm smoke roots: {elapsed:.6f}s, {calls} provider probes")
    assert calls == 0
    assert state.work.directories_scanned == state.work.content_reads == 0
    assert elapsed < 2


def test_green_lock_parent_verdict_tracks_scope_and_unavailable_files(tmp_path):
    from contextlib import nullcontext

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n")
    (root / "other.py").write_text("VALUE = 1\n")
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root), target=root / "app.py",
                              editable_product_paths=("app.py", "missing.py"),
                              binding=lambda: {"product_revision": "revision"})
    write = lambda _: nullcontext()
    if hasattr(green_lock, "_workspace_consumers"):
        from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
        consumers = WorkspaceConsumers(state)
        bind_consumers(root, consumers)
        write = state.begin_write

    runtime.green_lock = green_lock.GreenLock(runtime, mode="on")
    assert green_lock.product_revision_change_from_parent(runtime)[::2] == (
        False, "product_file_digests_match_parent",
    )
    assert green_lock.admit_widened_editable_path(runtime, "other.py")
    with write(("other.py",)):
        (root / "other.py").write_text("VALUE = 2\n")
    assert green_lock.product_revision_change_from_parent(runtime)[::2] == (
        True, "product_file_digests_differ_from_parent",
    )
    with write(("app.py",)):
        (root / "app.py").unlink()
        (root / "app.py").symlink_to("other.py")
    assert green_lock.product_revision_change_from_parent(runtime)[::2] == (
        False, "parent_product_revision_unavailable",
    )


def test_smoke_root_cache_inputs(tmp_path, monkeypatch):
    from contextlib import nullcontext

    root = tmp_path / "workspace"
    root.mkdir()
    (root / "app.py").write_text("import dependency\n")
    for name in ("first", "second"):
        (root / name).mkdir()
    write = lambda _: nullcontext()
    if hasattr(smoke, "_workspace_consumers"):
        from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        state = WorkspaceState(WorkspaceFence.create(root, tmp_path / "fence"))
        consumers = WorkspaceConsumers(state)
        bind_consumers(root, consumers)
        write = state.begin_write

    monkeypatch.setenv("PYTHONPATH", "first")
    paths = ("app.py",)
    assert smoke.source_roots(root, paths) == (str(root), str(root / "first"))
    first_digest = smoke.tree_digest(root, paths)
    monkeypatch.setenv("PYTHONPATH", "second")
    assert smoke.source_roots(root, paths) == (str(root), str(root / "second"))
    with write(("app.py", "pytest.ini")):
        (root / "app.py").write_text("VALUE = 2\n")
        (root / "pytest.ini").write_text("[pytest]\npythonpath = first\n")
    assert smoke.tree_digest(root, paths) != first_digest
    assert smoke.source_roots(root, paths) == (
        str(root), str(root / "first"), str(root / "second"),
    )


def test_foreground_shell_uses_distinct_pre_call_and_protected_sources(tmp_path):
    from contextlib import nullcontext
    from kernel_exec_session import CleanupReport, ExecResult
    from kernel_shell_tool import ShellTool
    from kernel_tool_settings import ToolSettings
    from kernel_tool_tape import configure_replay, tape_for_run

    class ProcessBoundary:
        def configure(self, capability):
            pass

        def run(self, argv, capability, *, cwd):
            asset = cwd / "asset.bin"
            if argv[-1] == "python first.py":
                asset.write_bytes(b"\0forbidden")
                asset.chmod(0o755)
                (cwd / "test_app.py").write_bytes(b"assert False\n")
            else:
                asset.write_bytes(b"\0accepted")
            return ExecResult(0, False, b"ok\n", False, 3, 0.01,
                              CleanupReport(True, 0, (), "settled"))

    def run(name, *, trees):
        directory = tmp_path / name
        root = directory / "workspace"
        root.mkdir(parents=True)
        asset = root / "asset.bin"
        protected = root / "test_app.py"
        asset.write_bytes(b"\0parent")
        original_mode = asset.stat().st_mode & 0o777
        protected.write_bytes(b"assert True\n")
        run_dir = directory / "run"
        run_dir.mkdir()
        write = lambda _: nullcontext()
        if trees:
            from kernel_effects import scan
            from kernel_workspace_consumers import ShellWorkspace, WorkspaceConsumers, bind_consumers
            from kernel_workspace_objects import ObjectRef, open_workspace_objects
            from kernel_workspace_state import WorkspaceFence, WorkspaceState
            from io import BytesIO

            state = WorkspaceState(WorkspaceFence.create(root, directory / "fence"))
            store_parent = directory / "private"
            store_parent.mkdir(mode=0o700)
            store = open_workspace_objects(store_parent, "f" * 64, enabled=True)
            refs = {}

            class RollbackBlobs:
                def get(self, digest):
                    with store.lease() as lease:
                        return lease.read(refs[digest])

            class ShellPersistenceFixture(ShellWorkspace):
                # Exercise the production reconciliation using real E5 blob storage.
                # Root registration and tree publishing belong to the adjacent units.
                def baseline(self):
                    baseline = scan(root, exclude=lambda _: False)
                    self._save(baseline)
                    return baseline

                def _commit(self, boundary, snapshot, call_id, kind):
                    assert snapshot.validate_for_use(boundary)
                    self._save(self._manifest(snapshot.tree))

                def _save(self, manifest):
                    pending = []
                    for path, entry in manifest.items():
                        if entry.kind == "file" and entry.digest not in refs:
                            ref = ObjectRef(entry.digest, entry.size)
                            data = (root / path).read_bytes()
                            pending.append((ref, lambda data=data: BytesIO(data)))
                            refs[entry.digest] = ref
                    with store.lease() as lease:
                        lease.publish_many(pending)

            shell = ShellPersistenceFixture(state, store, parent_key=None)
            shell.blobs = RollbackBlobs()
            consumers = WorkspaceConsumers(state, shell=shell)
            bind_consumers(root, consumers)
            write = state.begin_write

        tool = ShellTool(run_dir, ToolSettings(run_env=False), {}, ProcessBoundary())
        tool.bind(root, {"files": {"asset.bin": ""}}, {str(protected)})
        configure_replay(run_dir, active=False)
        with write(("asset.bin", "test_app.py")):
            asset.write_bytes(b"\0precall")
            protected.write_bytes(b"assert 2\n")
        first = tool.execute({"command": "python first.py"}, "call-1")
        assert not first.applied
        assert asset.read_bytes() == b"\0precall"
        assert asset.stat().st_mode & 0o777 == original_mode
        assert protected.read_bytes() == b"assert True\n"
        second = tool.execute({"command": "python second.py"}, "call-2")
        assert asset.read_bytes() == b"\0accepted"
        assert len(second.applied) == 1 and second.applied[0].path == "asset.bin"
        tape = [tape_for_run().lookup(call) for call in ("call-1", "call-2")]
        return first, second, [(row.pre_manifest_digest, row.post_manifest_digest) for row in tape]

    legacy = run("legacy", trees=False)
    if hasattr(green_lock, "_workspace_consumers"):
        assert run("objects", trees=True) == legacy


def test_real_root_registration_and_snapshot_checkout(tmp_path, monkeypatch):
    if not hasattr(green_lock, "_workspace_consumers"):
        return
    from kernel_workspace_state import WorkspaceSnapshot
    if not hasattr(WorkspaceSnapshot, "persist_with_metadata"):
        return

    from kernel_workspace_consumers import bind_run_consumers, copy_pinned_workspace
    from kernel_workspace_refs import bind_run_store, read_root
    from kernel_workspace_tree_io import lookup_ref

    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    root = run / "workspace"
    root.mkdir()
    source = root / "asset.bin"
    source.write_bytes(b"\0before")
    (root / "ignored").mkdir()
    (root / "ignored" / "cache").write_bytes(b"cache")
    store = bind_run_store(run, "d" * 64)
    context = bind_run_consumers(run, root, parent=root)
    shell = context.shell
    original = shell.baseline()
    tree, before = shell.before("call-1")
    assert before == original
    with shell.execution(SimpleNamespace(sandboxed=True, workspace_writable=True)):
        source.write_bytes(b"\0after")
    from kernel_effects import WorkspaceEffectPolicy

    policy = WorkspaceEffectPolicy(lambda _: False, None, lambda _: True)
    settled = shell.settle(tree, before, policy, "call-1")
    assert [(entry.path, entry.change) for entry in settled.applied] == [
        ("asset.bin", "modified"),
    ]
    with store.lease() as lease:
        old = read_root(lease, shell._key("shell-pre", "call-1"))
        new = read_root(lease, shell._key("shell-post", "call-1"))
        assert old is not None and new is not None and old != new
        assert lookup_ref(lease, old.oid, (b"asset.bin",)) != lookup_ref(
            lease, new.oid, (b"asset.bin",))
    target = run / "checkout"
    assert copy_pinned_workspace(root, target, ignore=lambda _, names: {"ignored"}.intersection(names))
    assert not (target / "ignored").exists()
    assert (target / "asset.bin").stat().st_mtime_ns == source.stat().st_mtime_ns
    (target / "asset.bin").write_bytes(b"new checkout bytes")
    assert source.read_bytes() == b"\0after"
