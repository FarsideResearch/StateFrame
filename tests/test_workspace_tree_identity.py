"""Workspace identity and structural work budgets, exercised without a model."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import random
import stat
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pytest


def _modules():
    return (
        importlib.import_module("kernel_workspace_tree"),
        importlib.import_module("kernel_workspace_index"),
    )


def _legacy_snapshot(root: Path) -> tuple[str, int, int]:
    from kernel_verdict_identity import workspace_tree_hash

    reads = size = 0
    read_bytes = Path.read_bytes

    def counted(path: Path) -> bytes:
        nonlocal reads, size
        data = read_bytes(path)
        reads += 1
        size += len(data)
        return data

    with patch.object(Path, "read_bytes", counted):
        digest = workspace_tree_hash(root)
    return digest, reads, size


def test_sparse_edit_work_budget(tmp_path):
    payload = b"a" * 1024
    changed = b"b" * len(payload)
    for number in range(5000):
        (tmp_path / f"source_{number:05d}.py").write_bytes(payload)
    name = "source_02500.py"
    path = tmp_path / name

    if importlib.util.find_spec("kernel_workspace_index") is None:
        # The archived release has no index. Measure its actual workspace reader
        # so the budget goes red for full-tree work even before the new API exists.
        old, _, _ = _legacy_snapshot(tmp_path)
        path.write_bytes(changed)
        start = perf_counter()
        new, reads, size = _legacy_snapshot(tmp_path)
        elapsed = perf_counter() - start
        pages = scans = read_calls = scandir_calls = diff_pages = None
    else:
        tree, index_module = _modules()
        index = index_module.open_workspace_index(tmp_path, enabled=True)
        assert index is not None
        before = index.refresh()
        old = before.digest
        read_calls = scandir_calls = 0
        read = os.read
        scandir = os.scandir

        def counted_read(fd, size):
            nonlocal read_calls
            read_calls += 1
            return read(fd, size)

        def counted_scandir(fd):
            nonlocal scandir_calls
            scandir_calls += 1
            return scandir(fd)

        path.write_bytes(changed)
        with patch.object(os, "read", counted_read), patch.object(os, "scandir", counted_scandir):
            start = perf_counter()
            snapshot = index.refresh([os.fsencode(name)])
            elapsed = perf_counter() - start
        new = snapshot.digest
        work = index.work
        reads, size = work.content_reads, work.rehashed_bytes
        pages, scans = work.pages_visited, work.directories_scanned
        diff_work = tree.TreeWork()
        assert [change.path for change in tree.diff(before, snapshot, diff_work)] == [(os.fsencode(name),)]
        diff_pages = diff_work.pages_visited
        assert tree.lookup(snapshot, (os.fsencode(name),)).sha256 == hashlib.sha256(changed).digest()

    print(
        f"sparse edit: {elapsed:.6f}s reads={reads} bytes={size} pages={pages} scans={scans} "
        f"read_calls={read_calls} scandir_calls={scandir_calls} diff_pages={diff_pages}"
    )
    assert new != old
    assert (reads, size) == (1, len(changed))
    assert pages is not None and 0 < pages <= 48
    assert diff_pages is not None and 0 < diff_pages <= 48
    assert read_calls == 2
    assert scans == scandir_calls == 0
    assert elapsed < 0.050


def _truth(tree, directory: Path):
    """Independent exhaustive snapshot. Collect raw names and read every file."""
    items = []
    for path in directory.iterdir():
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            entry = tree.Symlink(mode, os.readlink(os.fsencode(path)))
        elif stat.S_ISDIR(info.st_mode):
            entry = _truth(tree, path)
        elif stat.S_ISREG(info.st_mode):
            data = path.read_bytes()
            entry = tree.File(mode, len(data), hashlib.sha256(data).digest(), info.st_nlink > 1)
        else:
            entry = tree.Special(mode, stat.S_IFMT(info.st_mode), info.st_rdev, info.st_size)
        items.append((os.fsencode(path.name), entry))
    return tree.directory(stat.S_IMODE(directory.stat().st_mode), reversed(items))


def _flatten(tree, root, prefix=()):
    if isinstance(root, tree.Directory):
        yield prefix, (type(root).__name__, root.mode)
        for name, entry in tree.children(root):
            yield from _flatten(tree, entry, (*prefix, name))
    else:
        yield prefix, (type(root).__name__, root.digest)


@pytest.mark.parametrize("seed", [5, 73, 182])
def test_sparse_history_matches_fresh_scan_and_diff(tmp_path, seed):
    tree, index_module = _modules()
    rng = random.Random(seed)
    for dirname in ("alpha", "beta", "empty"):
        (tmp_path / dirname).mkdir()
    for i in range(35):
        (tmp_path / "alpha" / f"shared-prefix-{i:02}").write_bytes(b"same")
    index = index_module.WorkspaceIndex(tmp_path)
    old = index.refresh()
    for step in range(100):
        base = rng.choice(["alpha", "beta"])
        name = f"shared-prefix-{rng.randrange(45):02}"
        path = tmp_path / base / name
        touched = [os.fsencode(f"{base}/{name}")]
        operation = rng.randrange(8)
        if step % 19 == 0:
            source, destination = ("alpha", "moved"), ("beta", "moved")
            if (tmp_path / source[0] / source[1]).exists():
                source, destination = destination, source
            if (tmp_path / destination[0] / destination[1]).exists():
                (tmp_path / destination[0] / destination[1]).rename(tmp_path / source[0] / source[1])
            else:
                (tmp_path / source[0] / source[1]).mkdir()
                (tmp_path / source[0] / source[1] / "file").write_bytes(b"nested")
            touched = [os.fsencode("/".join(parts)) for parts in (source, destination)]
        elif operation == 0:
            path.unlink(missing_ok=True)
        elif operation == 1 and not path.is_symlink() and path.is_file():
            path.chmod(rng.choice([0o600, 0o644, 0o755]))
        elif operation == 2 and not path.is_symlink() and path.is_file():
            os.utime(path, ns=(1000, 2000))
        elif operation == 3:
            path.unlink(missing_ok=True)
            path.symlink_to(rng.choice(["missing", "../empty", "../../outside"]))
        elif operation == 4 and os.path.lexists(path):
            other = "beta" if base == "alpha" else "alpha"
            path.rename(tmp_path / other / name)
            touched.append(os.fsencode(f"{other}/{name}"))
        else:
            path.unlink(missing_ok=True)
            path.write_bytes(rng.choice([b"same", b"", rng.randbytes(12)]))

        current = index.refresh(touched)
        truth = _truth(tree, tmp_path)
        assert current.digest == truth.digest
        before = dict(_flatten(tree, old))
        after = dict(_flatten(tree, truth))
        expected = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        assert [change.path for change in tree.diff(old, current)] == expected
        old = current


def test_kinds_modes_raw_names_and_touch(tmp_path):
    tree, index_module = _modules()
    (tmp_path / "empty").mkdir()
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "link").symlink_to("../missing")
    (tmp_path / "file").write_bytes(b"same")
    (tmp_path / "duplicate").write_bytes(b"same")
    raw_name = b"\xff"
    with open(os.path.join(os.fsencode(tmp_path), raw_name), "wb") as stream:
        stream.write(b"raw")
    index = index_module.WorkspaceIndex(tmp_path)
    first = index.refresh()
    assert first.digest == _truth(tree, tmp_path).digest
    assert [name for name, _ in tree.children(first)] == [b"duplicate", b"empty", b"fifo", b"file", b"link", raw_name]
    assert tree.lookup(first, (b"duplicate",)).digest == tree.lookup(first, (b"file",)).digest
    assert tree.lookup(first, (b"fifo",)).kind == stat.S_IFIFO
    assert tree.lookup(first, (b"link",)).target == b"../missing"
    assert tree.lookup(first, (b"link", b"child")) is None
    os.utime(tmp_path / "file", ns=(1000, 2000))
    assert index.refresh([b"file"]).digest == first.digest
    (tmp_path / "empty").chmod(0o711)
    assert [change.path for change in tree.diff(first, index.refresh([b"empty"]))] == [(b"empty",)]
    (tmp_path / "duplicate").unlink()
    os.link(tmp_path / "file", tmp_path / "duplicate")
    linked = index.refresh([b"file", b"duplicate"])
    assert linked.digest == _truth(tree, tmp_path).digest
    assert tree.lookup(linked, (b"file",)).hardlinked
    # Replace a symlink with a directory, then delete it with all descendants.
    (tmp_path / "link").unlink()
    (tmp_path / "link").mkdir()
    (tmp_path / "link" / "child").write_bytes(b"nested")
    directory = index.refresh([b"link"])
    assert [change.path for change in tree.diff(linked, directory)] == [(b"link",), (b"link", b"child")]
    (tmp_path / "link" / "child").unlink()
    (tmp_path / "link").rmdir()
    deleted = index.refresh([b"link"])
    assert [change.path for change in tree.diff(directory, deleted)] == [(b"link",), (b"link", b"child")]


def test_canonical_pages_across_split_collapse_and_order():
    tree, _ = _modules()
    file = tree.File(0o644, 1, hashlib.sha256(b"x").digest())
    rows = [(b"common-prefix-", file), *((f"common-prefix-{i:04}".encode(), file) for i in range(513))]
    expected = tree.directory(0o755, rows)
    rng = random.Random(11)
    rng.shuffle(rows)
    actual = tree.directory(0o755, [])
    for name, entry in rows:
        actual = tree.replace(actual, name, entry)
    assert actual.digest == expected.digest
    for name, _ in rows[:-2]:
        actual = tree.replace(actual, name, None)
    assert actual.digest == tree.directory(0o755, rows[-2:]).digest
    assert list(tree.diff(expected, tree.directory(0o755, reversed(rows)))) == []
    with pytest.raises(ValueError, match="duplicate"):
        tree.directory(0o755, rows[-2:] * 2)
    for name in (b"", b".", b"..", b"x/y", b"x\0y", b"/absolute"):
        with pytest.raises(ValueError):
            tree.replace(actual, name, file)


def test_versioned_encoding_golden():
    tree, _ = _modules()
    file = tree.File(0o644, 1, hashlib.sha256(b"x").digest())
    link = tree.Symlink(0o777, b"../\xff")
    assert file.digest.hex() == "a64ceae134de397fed84183398b6c301e2c258b42e855d2874115757ba91031b"
    assert link.digest.hex() == "a56aab090f3c026befdca10a2e56b64a0768fad9dc9e4be8d9597d731ba6c732"
    empty = tree.directory(0o755, [])
    assert empty.digest.hex() == "7ebeee4dc5dcf77c7e072cc5dd3fb94aadb9acc719ffcb8a2553e52a00ac3250"
    parent = tree.directory(0o755, [(b"\xff", link), (b"x", file)])
    assert parent.digest.hex() == "a9fa3be5adad693a0b7082e8304a28d90b2f4ce06703475d9aa65b45f722a5a6"


def test_index_defaults_off(tmp_path, monkeypatch):
    _, index_module = _modules()

    def unexpected(*args, **kwargs):
        pytest.fail("off must not construct an index")

    monkeypatch.setattr(index_module, "WorkspaceIndex", unexpected)
    assert index_module.open_workspace_index(tmp_path) is None


def test_stat_cache_rejects_racy_hits_and_dirty_paths(tmp_path, monkeypatch):
    _, index_module = _modules()
    path = tmp_path / "source"
    path.write_bytes(b"first")
    now = max(path.stat().st_ctime_ns, path.stat().st_mtime_ns)
    monkeypatch.setattr(index_module, "time_ns", lambda: now)
    index = index_module.WorkspaceIndex(tmp_path, trust_stat=True)
    original = index.refresh().digest
    assert index.refresh().digest == original
    assert index.work.content_reads == 1 and index.work.stat_hits == 0
    now += 3_000_000_000
    assert index.refresh().digest == original
    assert index.work.content_reads == 1
    assert index.refresh().digest == original
    assert index.work.content_reads == 0 and index.work.stat_hits == 1
    assert index.refresh([b"source"]).digest == original
    assert index.work.content_reads == 1 and index.work.stat_hits == 0
    assert index.refresh([]).digest == original
    assert index.work.content_reads == index.work.pages_visited == index.work.directories_scanned == 0
    now -= 1
    assert index.refresh().digest == original
    assert index.work.content_reads == 1 and index.work.stat_hits == 0
    previous = path.stat()
    path.write_bytes(b"other")
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert index.refresh().digest != original
    assert index.work.content_reads == 1


def test_failed_sparse_update_keeps_previous_snapshot(tmp_path):
    tree, index_module = _modules()
    (tmp_path / "first").write_bytes(b"old")
    (tmp_path / "link").symlink_to("missing")
    index = index_module.WorkspaceIndex(tmp_path)
    previous = index.refresh()
    (tmp_path / "first").write_bytes(b"new")
    with pytest.raises(OSError):
        index.refresh([b"first", b"link/child"])
    assert index.root is previous
    assert tree.lookup(previous, (b"first",)).sha256 == hashlib.sha256(b"old").digest()
    assert index.refresh([b"first"]).digest == _truth(tree, tmp_path).digest
