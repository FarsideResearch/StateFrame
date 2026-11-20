"""Exercise an opt-in workspace feed against the portable stat index."""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from statistics import median
from time import perf_counter

import pytest

from kernel_workspace_index import WorkspaceIndex


def test_clean_boundary_work_budget(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("clean feed budget requires Linux inotify")

    for number in range(50):
        directory = tmp_path / f"package_{number:02d}"
        directory.mkdir()
        for file_number in range(100):
            (directory / f"source_{file_number:03d}.py").write_bytes(b"answer = 42\n")

    start = perf_counter()
    index = WorkspaceIndex(tmp_path)
    feed = None
    if importlib.util.find_spec("kernel_workspace_feed") is not None:
        module = importlib.import_module("kernel_workspace_feed")
        feed = module.WorkspaceFeed(tmp_path, enabled=True)
        assert feed.rebuild(writer_fenced=True)
    expected = index.refresh().digest
    if feed is not None:
        assert not feed.poll(writer_fenced=True, after_full_sweep=True).requires_sweep
    cold = perf_counter() - start

    counts = dict(scandir=0, stat=0, read=0)

    def count(name):
        original = getattr(os, name)

        def counted(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(os, name, counted)

    for name in counts:
        count(name)

    durations = []
    content_reads = 0
    turns = 200
    try:
        for _ in range(turns):
            start = perf_counter()
            update = None if feed is None else feed.poll(writer_fenced=True)
            # Pre-feed E2 gets the complete tree from the same stat implementation
            # as fallback, allowing the test to measure old work before the API exists.
            paths = None if update is None or update.requires_sweep else update.paths
            actual = index.refresh(paths).digest
            durations.append(perf_counter() - start)
            content_reads += index.work.content_reads
            assert actual == expected
    finally:
        if feed is not None:
            feed.close()

    print(
        f"{turns} clean boundaries, 5000 files: cold={cold:.6f}s "
        f"total={sum(durations):.6f}s median={median(durations):.6f}s "
        f"p95={sorted(durations)[int(turns * 0.95) - 1]:.6f}s max={max(durations):.6f}s "
        f"scandir={counts['scandir']} stat={counts['stat']} read={counts['read']} "
        f"content_reads={content_reads}"
    )
    assert (counts["scandir"], content_reads) == (0, 0), (counts, content_reads)
    assert counts["stat"] <= 2 * turns, counts
    assert counts["read"] <= 2 * turns, counts
    assert max(durations) < 0.05, durations


@pytest.fixture
def feed_module():
    return pytest.importorskip("kernel_workspace_feed")


@pytest.fixture
def watched(tmp_path, feed_module):
    if sys.platform != "linux":
        pytest.skip("requires Linux inotify")
    root = tmp_path / "workspace"
    root.mkdir()
    feed = feed_module.WorkspaceFeed(root, enabled=True)
    try:
        yield root, feed
    finally:
        feed.close()


def _baseline(root, feed):
    assert feed.rebuild(writer_fenced=True)
    assert feed.poll(writer_fenced=True).sweep_reason == "baseline_required"
    index = WorkspaceIndex(root)
    index.refresh()
    update = feed.poll(writer_fenced=True, after_full_sweep=True)
    assert not update.requires_sweep
    assert update.paths == ()
    return index


def _reconcile(index, root, update):
    paths = None if update.requires_sweep else update.paths
    actual = index.refresh(paths)
    assert actual.digest == WorkspaceIndex(root).refresh().digest
    return actual


def test_disabled_and_unsupported_use_stat_truth(tmp_path, monkeypatch, feed_module):
    path = tmp_path / "source.py"
    path.write_bytes(b"before")
    index = WorkspaceIndex(tmp_path)
    before = index.refresh().digest

    def unexpected():
        pytest.fail("opened an inotify instance with the feed disabled or unsupported")

    monkeypatch.setattr(feed_module, "_open_inotify", unexpected)
    disabled = feed_module.WorkspaceFeed(tmp_path)
    assert not disabled.rebuild(writer_fenced=True)
    assert disabled.poll(writer_fenced=True).sweep_reason == "disabled"
    path.write_bytes(b"after!")
    assert _reconcile(index, tmp_path, disabled.poll()).digest != before

    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "darwin")
        unsupported = feed_module.WorkspaceFeed(tmp_path, enabled=True)
        assert not unsupported.rebuild(writer_fenced=True)
        update = unsupported.poll(writer_fenced=True)
    assert update.sweep_reason == "unsupported"
    path.write_bytes(b"again!")
    _reconcile(index, tmp_path, update)


def test_real_events_match_full_refresh(watched):
    root, feed = watched
    left = root / "left"
    right = root / "right"
    left.mkdir()
    right.mkdir()
    source = left / "source.py"
    source.write_bytes(b"old")
    index = _baseline(root, feed)
    before = index.root.digest

    source.write_bytes(b"new")
    os.utime(source, ns=(1_000_000_000, 1_000_000_000))
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"left/source.py",)
    assert _reconcile(index, root, update).digest != before
    assert index.work.content_reads == 1
    assert index.work.directories_scanned == 0

    source.chmod(0o700)
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"left/source.py",)
    _reconcile(index, root, update)

    target = right / "source.py"
    source.rename(target)
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"left/source.py", b"right/source.py")
    _reconcile(index, root, update)

    raw = b"raw_\xff.py"
    replacement = right / os.fsdecode(raw)
    replacement.write_bytes(b"replace")
    replacement.replace(target)
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"right/" + raw, b"right/source.py")
    _reconcile(index, root, update)

    target.unlink()
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"right/source.py",)
    _reconcile(index, root, update)


def test_move_in_populated_tree_and_watch_new_descendants(watched):
    root, feed = watched
    index = _baseline(root, feed)
    outside = root.parent / "outside"
    (outside / "nested" / "deep").mkdir(parents=True)
    (outside / "nested" / "deep" / "source.py").write_bytes(b"old")

    incoming = root / "incoming"
    outside.rename(incoming)
    update = feed.poll(writer_fenced=True)
    assert not update.requires_sweep
    assert update.paths == (b"incoming",)
    observed = _reconcile(index, root, update)

    (incoming / "nested" / "deep" / "source.py").write_bytes(b"new")
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"incoming/nested/deep/source.py",)
    assert _reconcile(index, root, update).digest != observed.digest

    (root / "created" / "nested").mkdir(parents=True)
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"created",)
    _reconcile(index, root, update)
    (root / "created" / "nested" / "file").touch()
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"created/nested/file",)
    _reconcile(index, root, update)


def test_directory_rename_falls_back_and_rearms(watched):
    root, feed = watched
    directory = root / "old"
    directory.mkdir()
    (directory / "source.py").write_bytes(b"old")
    index = _baseline(root, feed)

    moved = root / "new"
    directory.rename(moved)
    update = feed.poll(writer_fenced=True)
    assert update.sweep_reason in {"directory_departed", "watch_lost"}
    assert update.paths == ()
    _reconcile(index, root, update)
    index = _baseline(root, feed)
    (moved / "source.py").write_bytes(b"new")
    update = feed.poll(writer_fenced=True)
    assert update.paths == (b"new/source.py",)
    _reconcile(index, root, update)


def test_deleted_watch_falls_back(watched):
    root, feed = watched
    directory = root / "empty"
    directory.mkdir()
    index = _baseline(root, feed)
    directory.rmdir()
    update = feed.poll(writer_fenced=True)
    assert update.sweep_reason in {"directory_departed", "watch_lost"}
    _reconcile(index, root, update)


def test_coverage_prunes_directories_and_never_follows_symlinks(tmp_path, feed_module):
    if sys.platform != "linux":
        pytest.skip("requires Linux inotify")
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    skipped = root / "skip"
    skipped.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    visited = []

    def include(path):
        visited.append(path)
        return path != b"skip"

    feed = feed_module.WorkspaceFeed(root, enabled=True, include_directory=include)
    try:
        _baseline(root, feed)
        assert visited == [b"skip"]
        (skipped / "file").touch()
        (outside / "file").touch()
        assert feed.poll(writer_fenced=True).paths == ()
        assert not feed.poll(writer_fenced=True).requires_sweep
    finally:
        feed.close()


def test_startup_acknowledgement_keeps_queued_events(watched):
    root, feed = watched
    assert feed.rebuild(writer_fenced=True)
    (root / "source.py").write_bytes(b"startup")
    for _ in range(2):
        assert feed.poll(writer_fenced=True).sweep_reason == "baseline_required"
    index = WorkspaceIndex(root)
    index.refresh()
    update = feed.poll(writer_fenced=True, after_full_sweep=True)
    assert update.paths == (b"source.py",)
    _reconcile(index, root, update)
    assert feed.poll(writer_fenced=True).paths == ()


@pytest.mark.parametrize(("event", "reason"), [
    ("overflow", "overflow"),
    ("ignored", "watch_lost"),
    ("unmount", "watch_lost"),
    ("unknown_watch", "unknown_watch"),
    ("truncated", "malformed_event"),
    ("interrupted", "drain_failed"),
])
def test_queue_loss_discards_partial_batch(watched, monkeypatch, event, reason):
    import errno
    import struct

    root, feed = watched
    source = root / "source.py"
    source.write_bytes(b"before")
    index = _baseline(root, feed)
    source.write_bytes(b"after!")
    fd = feed._fd
    read = os.read
    wd = next(iter(feed._watches))
    packet = struct.Struct("=iIII")
    events = {
        "overflow": packet.pack(-1, 0x4000, 0, 0),
        "ignored": packet.pack(wd, 0x8000, 0, 0),
        "unmount": packet.pack(wd, 0x2000, 0, 0),
        "unknown_watch": packet.pack(wd + 100, 0x0004, 0, 0),
        "truncated": packet.pack(wd, 0x0002, 0, 16) + b"partial",
    }
    calls = 0

    def inject(descriptor, size):
        nonlocal calls
        if descriptor != fd:
            return read(descriptor, size)
        calls += 1
        if calls == 1:
            return read(descriptor, size)  # Real dirty event before the failure.
        if event == "interrupted":
            raise InterruptedError(errno.EINTR, "interrupted")
        return events[event]

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", inject)
        update = feed.poll(writer_fenced=True)
    assert calls == 2
    assert update.sweep_reason == reason
    assert update.paths == ()
    assert feed.poll(writer_fenced=True, after_full_sweep=True) == update
    with pytest.raises(OSError) as exc:
        os.fstat(fd)
    assert exc.value.errno == errno.EBADF
    _reconcile(index, root, update)
    _baseline(root, feed)


@pytest.mark.parametrize("failure", ["limit", "reused_descriptor"])
def test_partial_watch_setup_cannot_be_acknowledged(watched, feed_module, monkeypatch, failure):
    import errno

    root, feed = watched
    (root / "child").mkdir()
    real_open = feed_module._open_inotify
    seen = []

    def limited_open():
        fd, watch = real_open()
        seen.append(fd)
        descriptors = []

        def limited_watch(directory_fd):
            if not descriptors:
                descriptors.append(watch(directory_fd))
                return descriptors[0]
            if failure == "limit":
                raise OSError(errno.ENOSPC, "watch limit")
            return descriptors[0]

        return fd, limited_watch

    with monkeypatch.context() as patch:
        patch.setattr(feed_module, "_open_inotify", limited_open)
        assert not feed.rebuild(writer_fenced=True)
    update = feed.poll(writer_fenced=True, after_full_sweep=True)
    assert update.sweep_reason == ("watch_setup_failed" if failure == "limit" else "watch_reused")
    with pytest.raises(OSError) as exc:
        os.fstat(seen[0])
    assert exc.value.errno == errno.EBADF
    _baseline(root, feed)


def test_root_replaced_invalidates_old_generation(watched):
    root, feed = watched
    _baseline(root, feed)
    root.rename(root.parent / "old-root")
    root.mkdir()
    (root / "new").touch()
    update = feed.poll(writer_fenced=True)
    assert update.sweep_reason in {"root_changed", "watch_lost"}
    assert update.paths == ()
    reopened = _baseline(root, feed)
    assert reopened.root.digest == WorkspaceIndex(root).refresh().digest


def test_unfenced_mmap_write_uses_stat_truth(watched):
    import mmap

    root, feed = watched
    path = root / "mapped"
    path.write_bytes(b"before")
    index = _baseline(root, feed)
    before = index.root.digest
    with path.open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as data:
        data[:] = b"after!"
        data.flush()
        update = feed.poll()
        assert update.sweep_reason == "unfenced"
        assert _reconcile(index, root, update).digest != before
    assert feed.poll(writer_fenced=True).sweep_reason == "unfenced"
    _baseline(root, feed)


def test_crash_and_reopen_requires_new_stat_baseline(watched):
    import signal

    root, feed = watched
    path = root / "source.py"
    path.write_bytes(b"old")
    index = WorkspaceIndex(root)
    before = index.refresh().digest
    script = """
import os
import signal
import sys
from kernel_workspace_feed import WorkspaceFeed
feed = WorkspaceFeed(sys.argv[1], enabled=True)
assert feed.rebuild(writer_fenced=True)
assert not os.get_inheritable(feed._fd)
assert not os.get_inheritable(feed._root_fd)
os.kill(os.getpid(), signal.SIGKILL)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(root)], timeout=10, check=False,
    )
    assert process.returncode == -signal.SIGKILL
    path.write_bytes(b"new")  # Changed while neither watch generation was alive.
    assert feed.rebuild(writer_fenced=True)
    assert feed.poll(writer_fenced=True).sweep_reason == "baseline_required"
    assert index.refresh().digest != before
    assert not feed.poll(writer_fenced=True, after_full_sweep=True).requires_sweep


def test_fork_cannot_drain_the_owners_queue(watched):
    root, _ = watched
    script = """
import os
import sys
from pathlib import Path
from kernel_workspace_feed import WorkspaceFeed
from kernel_workspace_index import WorkspaceIndex

root = Path(sys.argv[1])
feed = WorkspaceFeed(root, enabled=True)
assert feed.rebuild(writer_fenced=True)
WorkspaceIndex(root).refresh()
assert not feed.poll(writer_fenced=True, after_full_sweep=True).requires_sweep
(root / "source.py").write_bytes(b"queued")
pid = os.fork()
if pid == 0:
    update = feed.poll(writer_fenced=True)
    os._exit(0 if update.sweep_reason == "forked" else 1)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
update = feed.poll(writer_fenced=True)
assert not update.requires_sweep and update.paths == (b"source.py",)
feed.close()
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(root)], timeout=10, capture_output=True, check=False,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
