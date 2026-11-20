"""Workspace observation cost and freshness at real host boundaries."""
from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import os
from collections import Counter
from time import perf_counter
from types import SimpleNamespace

import pytest

def test_certified_clean_boundaries_do_no_workspace_walk(tmp_path, monkeypatch):
    from kernel_live_projection import WorkspaceViewStore

    root = tmp_path / "workspace"
    root.mkdir()
    for directory_id in range(50):
        directory = root / f"pkg_{directory_id:02d}"
        directory.mkdir()
        for file_id in range(100):
            (directory / f"module_{file_id:02d}.py").write_bytes(b"VALUE = 1\n")

    options = {}
    if importlib.util.find_spec("kernel_workspace_state") is not None:
        from kernel_workspace_state import WorkspaceFence, WorkspaceState

        fence = WorkspaceFence.create(root, tmp_path / "host")
        options["workspace_state"] = WorkspaceState(fence)
    store = WorkspaceViewStore(root, "fixture", **options)
    started = perf_counter()
    initial = store.current_view()
    cold_seconds = perf_counter() - started
    assert len(initial.product_files) == 5_000

    counts = Counter()

    def count(owner, name):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapper)

    count(os, "scandir")
    count(os, "lstat")
    count(os, "stat")
    count(os, "read")
    count(hashlib, "file_digest")
    times = []
    for turn in range(200):
        started = perf_counter()
        current = store.current_view("external" if turn % 2 else "test")
        times.append(perf_counter() - started)
        assert current is initial
    print(f"cold={cold_seconds:.6f}s, warm total={sum(times):.6f}s, "
          f"warm max={max(times):.6f}s, counts={dict(counts)}")
    assert counts["scandir"] == counts["read"] == counts["file_digest"] == 0, counts
    # Config discovery may probe fixed paths and their ancestors. This budget
    # scales with the root depth, independent of how many files are in it.
    assert counts["stat"] + counts["lstat"] <= 200 * (8 * len(root.parts) + 24), counts
    assert max(times) < 0.050, times


def _write_equal_size(path, value):
    before = path.stat()
    path.write_bytes(value)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def _external_writer(fence, ready, release, crash):
    with fence.writer():
        (fence.root / "old").rename(fence.root / "new")
        _write_equal_size(fence.root / "new" / "value.py", b"V = 2\n")
        ready.set()
        if not release.wait(15):
            raise TimeoutError("test writer did not receive release")
        if crash:
            os._exit(19)


@pytest.fixture
def workspace(tmp_path):
    state_module = pytest.importorskip("kernel_workspace_state")
    root = tmp_path / "ancestor" / "workspace"
    (root / "old").mkdir(parents=True)
    (root / "old" / "value.py").write_bytes(b"V = 1\n")
    return root, state_module.WorkspaceState(
        state_module.WorkspaceFence.create(root, tmp_path / "host"))


def test_backdated_edit_and_noop_writer_advance_observation(workspace):
    from kernel_live_projection import WorkspaceViewStore, workspace_binding

    root, state = workspace
    store = WorkspaceViewStore(root, "fixture", workspace_state=state)
    initial = store.current_view()
    path = root / "old" / "value.py"
    mtime = path.stat().st_mtime_ns
    with state.begin_write(("old/value.py",)):
        _write_equal_size(path, b"V = 2\n")
    current = store.current_view()
    assert current.as_binding() == workspace_binding(root, "fixture")
    assert current.product_revision != initial.product_revision
    assert current.mutation_epoch > initial.mutation_epoch
    assert path.stat().st_mtime_ns == mtime
    assert state.work.directories_scanned == 0
    assert state.work.content_reads == 1

    with state.begin_write(()):
        pass
    noop = store.current_view()
    assert noop.product_revision == current.product_revision
    assert noop.mutation_epoch > current.mutation_epoch
    assert state.work.directories_scanned == state.work.content_reads == 0


def test_writer_exception_invalidates_even_an_empty_footprint(workspace):
    root, state = workspace
    with state.pin("before") as initial:
        assert initial.validate_for_use("before")
        assert not initial.validate_for_use("somewhere else")
    assert not initial.validate_for_use("before")

    with pytest.raises(ValueError, match="failed"):
        with state.begin_write(()):
            _write_equal_size(root / "old" / "value.py", b"V = 2\n")
            raise ValueError("failed")
    with state.pin("after") as current:
        assert current.tree.digest != initial.tree.digest
        assert state.work.directories_scanned == 2
        assert state.work.content_reads == 1


def test_reported_mutation_revokes_an_active_snapshot(workspace):
    _, state = workspace
    with state.pin("read") as snapshot:
        assert snapshot.validate_for_use("read")
        state.invalidate()
        assert not snapshot.validate_for_use("read")


@pytest.mark.parametrize("crash", [False, True])
def test_cross_process_writer_blocks_pin_and_reconciles_after_exit(workspace, crash):
    from kernel_live_projection import WorkspaceViewStore, workspace_binding
    from kernel_workspace_state import WorkspaceUnavailable

    root, state = workspace
    store = WorkspaceViewStore(root, "fixture", workspace_state=state)
    initial = store.current_view()
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    writer = ctx.Process(target=_external_writer, args=(state.fence, ready, release, crash))
    with state.pin("read") as pin:
        with pytest.raises(WorkspaceUnavailable):
            with state.fence.writer():
                pytest.fail("writer entered under a reader lease")
        assert pin.validate_for_use("read")
    writer.start()
    try:
        assert ready.wait(10), writer.exitcode
        with pytest.raises(WorkspaceUnavailable):
            store.current_view()
    finally:
        release.set()
        writer.join(10)
        if writer.is_alive():
            writer.kill()
            writer.join(5)
    assert writer.exitcode == (19 if crash else 0)

    current = store.current_view()
    assert current.as_binding() == workspace_binding(root, "fixture")
    assert current.product_revision != initial.product_revision
    assert tuple(current.product_files) == ("new/value.py",)
    assert state.work.directories_scanned == 2


def test_root_replacement_invalidates_clean_token(workspace):
    from kernel_workspace_state import WorkspaceUnavailable

    root, state = workspace
    with state.pin("before") as old:
        assert old.validate_for_use("before")
    with state.begin_write(None):
        root.rename(root.with_name("retired"))
        root.mkdir()
        (root / "replacement.py").write_bytes(b"V = 9\n")
    with pytest.raises(WorkspaceUnavailable):
        with state.pin("after"):
            pytest.fail("a replaced root was accepted")


def test_unreadable_ancestor_invalidates_clean_token(workspace):
    from kernel_workspace_state import WorkspaceUnavailable

    root, state = workspace
    with state.pin("before"):
        pass
    previous_mode = root.parent.stat().st_mode & 0o777
    root.parent.chmod(0)
    try:
        with pytest.raises(WorkspaceUnavailable):
            with state.pin("after"):
                pytest.fail("an unreadable ancestor was accepted")
    finally:
        root.parent.chmod(previous_mode)
    with state.pin("recovered") as recovered:
        assert recovered.validate_for_use("recovered")


@pytest.mark.parametrize("age", [1, 3_000_000_000])
def test_opaque_stat_fallback_rehashes_identical_reported_signatures(workspace, monkeypatch, age):
    import kernel_workspace_index as index_module
    from kernel_workspace_state import WorkspaceState

    root, state = workspace
    state = WorkspaceState(state.fence, trust_stat=True)
    target = root / "old" / "value.py"
    real_signature = index_module._signature
    frozen = real_signature(target.stat())
    monkeypatch.setattr(index_module, "_signature", lambda info: (
        frozen if info.st_ino == frozen[1] and info.st_dev == frozen[0]
        else real_signature(info)))
    monkeypatch.setattr(index_module, "time_ns", lambda: max(frozen[3:5]) + age)
    with state.pin("before") as old:
        pass
    with state.fence.writer():  # a separate host reader will see an opaque epoch
        _write_equal_size(target, b"V = 2\n")
    with state.pin("after") as new:
        assert new.tree.digest != old.tree.digest
        assert state.work.content_reads == 1
        assert state.work.stat_hits == 0


def test_rename_during_scan_does_not_publish_partial_view(workspace, monkeypatch):
    from kernel_workspace_state import WorkspaceUnavailable

    root, state = workspace
    original = os.read
    moved = False

    def rename_while_reading(fd, count):
        nonlocal moved
        if not moved:
            moved = True
            (root / "old").rename(root / "new")
        return original(fd, count)

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", rename_while_reading)
        with pytest.raises(WorkspaceUnavailable):
            with state.pin("raced"):
                pytest.fail("the interrupted scan published a tree")
    with state.pin("retry") as pin:
        assert pin.validate_for_use("retry")
    assert moved


def test_fenced_projection_preserves_binding_and_filters_before_content_reads(workspace):
    from kernel_live_projection import WorkspaceViewStore, workspace_binding

    root, state = workspace
    (root / "tests").mkdir()
    (root / "tests" / "test_public.py").write_bytes(b"assert True\n")
    (root / "link.py").symlink_to("/unused-target")
    (root / "report.xml").write_text("test output", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "private").write_text("excluded", encoding="utf-8")
    store = WorkspaceViewStore(root, "fixture", workspace_state=state,
                               scope_identity="scope", policy_identity="policy",
                               authorization_scope="authorization", tenant_identity="tenant")
    command = ("pytest", "--junitxml=report.xml")
    checked = store.current_view(command=command, run_write_set=("report.xml",))
    assert checked.as_binding() == workspace_binding(root, "fixture", command=command)
    assert state.work.content_reads == 2
    assert checked.product_files["link.py"] == "symlink:/unused-target"
    assert checked.identity()["scope_identity"] == "scope"
    assert checked.identity()["policy_identity"] == "policy"
    assert checked.identity()["authorization_scope"] == "authorization"
    assert checked.identity()["tenant_identity"] == "tenant"

    unfiltered = store.current_view()
    assert "report.xml" in unfiltered.product_files
    store.update_context(environment="new environment")
    assert store.current_view().environment_revision == "new environment"
    with state.begin_write(("old/value.py",)):
        _write_equal_size(root / "old" / "value.py", b"V = 3\n")
    store.note_mutation(("old/value.py",), "edit")
    assert store.current_view().as_binding() == workspace_binding(root, "new environment")


def test_green_lock_parent_verdict_and_reason_match_legacy_binding(workspace):
    from kernel_green_lock import product_revision_change_from_parent
    from kernel_live_projection import WorkspaceViewStore, workspace_binding

    root, state = workspace
    store = WorkspaceViewStore(root, "fixture", workspace_state=state)
    starting = workspace_binding(root, "fixture")
    legacy = SimpleNamespace(starting_binding=starting, binding=lambda: workspace_binding(root, "fixture"))
    fenced = SimpleNamespace(starting_binding=starting, binding=lambda: store.current_view().as_binding())
    assert product_revision_change_from_parent(fenced) == product_revision_change_from_parent(legacy)
    assert not product_revision_change_from_parent(fenced)[0]
    with state.begin_write(("old/value.py",)):
        _write_equal_size(root / "old" / "value.py", b"V = 2\n")
    changed = product_revision_change_from_parent(fenced)
    assert changed[0]
    assert changed == product_revision_change_from_parent(legacy)
