"""Canonical workspace records and sparse CAS traversal."""
from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from io import BytesIO
from time import perf_counter
from types import ModuleType

import pytest

from stateframe.engine import kernel_effects as effects
import kernel_workspace_tree as tree
from kernel_workspace_objects import ObjectIntegrityError, ObjectRef, open_workspace_objects

codec = (importlib.import_module("kernel_workspace_tree_io")
         if importlib.util.find_spec("kernel_workspace_tree_io") is not None else None)


def registry_stub():
    """Provide the frozen Ref value type when testing before the registry lands."""
    if "kernel_workspace_refs" in sys.modules or importlib.util.find_spec("kernel_workspace_refs"):
        return None

    @dataclass(frozen=True)
    class Ref:
        kind: str
        oid: str

    registry = ModuleType("kernel_workspace_refs")
    registry.Ref = Ref
    return registry


@pytest.fixture(autouse=True)
def frozen_ref(monkeypatch):
    registry = registry_stub()
    if registry is not None:
        monkeypatch.setitem(sys.modules, registry.__name__, registry)


@pytest.fixture
def cas():
    if codec is None:
        pytest.skip("persisted workspace tree codec is unavailable")
    return MemoryLease()


class MemoryLease:
    """In-memory CAS implementing the frozen lease contract for work counts."""

    def __init__(self):
        self.objects = {}
        self.work = Counter()

    def publish_many(self, entries):
        for ref, source in entries:
            self.work["publications"] += 1
            with source() as stream:
                data = stream.read()
            assert (sha256(data).hexdigest(), len(data)) == (ref.digest, ref.size)
            self.objects[ref.digest] = data

    def resolve(self, oid, *, max_bytes):
        self.work["resolves"] += 1
        if oid not in self.objects:
            raise FileNotFoundError(oid)
        size = len(self.objects[oid])
        if size > max_bytes:
            raise ObjectIntegrityError("object exceeds bound")
        return ObjectRef(oid, size)

    def read(self, ref):
        self.work["reads"] += 1
        if ref.digest not in self.objects:
            raise FileNotFoundError(ref.digest)
        data = self.objects[ref.digest]
        if (sha256(data).hexdigest(), len(data)) != (ref.digest, ref.size):
            raise ObjectIntegrityError("object failed verification")
        return data

    def copy_verified(self, ref, output, *, max_bytes):
        assert ref.size <= max_bytes
        output.write(self.read(ref))


def test_sparse_persisted_diff_work_budget(monkeypatch):
    leaf = tree.File(0o644, 1, sha256(b"x").digest())
    changed = tree.File(0o644, 1, sha256(b"y").digest())
    before = tree.directory(0o755, (
        (f"file-{number:04}".encode(), leaf) for number in range(5000)
    ))
    after = tree.replace(tree.replace(before, b"file-0100", changed), b"file-4010", None)
    work = Counter()
    if codec is None:
        # The existing persistent boundary compares full path manifests.
        def manifest(root):
            return {
                name.decode(): effects.Entry("file", node.mode, node.sha256.hex(), node.size)
                for name, node in tree.children(root)
            }

        old, new = manifest(before), manifest(after)
        original = effects.Entry.__eq__

        def count_comparison(left, right):
            work["comparisons"] += 1
            return original(left, right)

        monkeypatch.setattr(effects.Entry, "__eq__", count_comparison)
        started = perf_counter()
        paths = [row.path.encode() for row in effects.diff(old, new)]
    else:
        lease = MemoryLease()
        old = codec.write_tree(lease, before)
        new = codec.write_tree(lease, after, previous=old)
        lease.work.clear()
        started = perf_counter()
        paths = [row.path[0] for row in codec.iter_changes(lease, old, new)]
        work = lease.work
    elapsed = perf_counter() - started
    print(f"sparse diff: {elapsed:.6f}s {dict(work)}")
    assert paths == [b"file-0100", b"file-4010"]
    assert max(work.values(), default=0) <= 96, dict(work)
    assert elapsed < 0.25


def _field(data):
    return len(data).to_bytes(4, "big") + data


def _raw(tag, *fields):
    return b"stateframe.workspace-tree/1\0" + b"".join(map(_field, (tag, *fields)))


def _uint(value):
    return value.to_bytes(8, "big")


def _store(lease, data):
    oid = sha256(data).hexdigest()
    lease.publish_many([(ObjectRef(oid, len(data)), partial(BytesIO, data))])
    return oid


def _digest(data):
    return sha256(data).digest()


def _dir(page):
    return _raw(b"dir", _uint(0o755), bytes.fromhex(page) if page else b"")


def test_exact_preimages_and_prefix_terminal(cas):
    mode = 0o755
    file = tree.File(0o644, 1, _digest(b"x"))
    link = tree.Symlink(0o777, b"../\xff")
    directory = tree.directory(0o711, ())
    names = [b"a"] + [b"a" + bytes((i,)) for i in range(128, 144)]
    entries = dict.fromkeys(names, file)
    entries[b"a\x80"] = link
    entries[b"a\x8f"] = directory

    file_raw = _raw(b"file", _uint(0o644), _uint(1), _digest(b"x"), b"\0")
    link_raw = _raw(b"symlink", _uint(0o777), b"../\xff")
    dir_raw = _raw(b"dir", _uint(0o711), b"")
    terminal = _field(b"a") + _digest(file_raw)
    records = [file_raw, link_raw, dir_raw]
    slots = []
    for name in names[1:]:
        child = {b"a\x80": link_raw, b"a\x8f": dir_raw}.get(name, file_raw)
        bucket = _raw(b"bucket", _field(name) + _digest(child))
        slots.append(name[1:] + _digest(bucket))
        records.append(bucket)
    inner = _raw(b"branch", terminal, *slots)
    outer = _raw(b"branch", b"", b"a" + _digest(inner))
    root = _raw(b"dir", _uint(mode), _digest(outer))
    records += [inner, outer, root]

    observed = tree.directory(mode, reversed(tuple(entries.items())))
    assert codec.write_tree(cas, observed) == _digest(root).hex() == observed.digest.hex()
    assert cas.objects == {_digest(record).hex(): record for record in records}
    assert [path for path, _, _ in codec.iter_entries(cas, observed.digest.hex())] == [
        (), *((name,) for name in names)
    ]
    before = codec.lookup_ref(cas, observed.digest.hex(), (b"a\x8f",))
    assert before.kind == "tree"
    assert codec.read_entry(cas, before).mode == 0o711
    assert codec.lookup_ref(cas, observed.digest.hex(), (b"a\x80", b"anything")) is None
    assert codec.lookup_ref(cas, observed.digest.hex(), (b"a\xff",)) is None


def test_changes_match_e2_on_split_collapse_and_type_changes(cas):
    file = tree.File(0o644, 1, _digest(b"x"))
    old = tree.directory(0o755, (
        (f"prefix-{idx}".encode(), file) for idx in range(17)
    ))
    old = tree.replace(old, b"prefix-0", tree.directory(0o755, [(b"inside", file)]))
    new = tree.with_mode(old, 0o750)
    new = tree.replace(new, b"prefix-0", file)
    new = tree.replace(new, b"prefix-1", None)
    new = tree.replace(new, b"prefix-9", tree.directory(0o700, [(b"inside", file)]))
    root = codec.write_tree(cas, old)
    changed = codec.write_tree(cas, new, previous=root)
    for before, after, base_oid, next_oid in [(old, new, root, changed), (new, old, changed, root)]:
        expected = [(r.path, r.before.digest.hex() if r.before else None,
                     r.after.digest.hex() if r.after else None) for r in tree.diff(before, after)]
        observed = [(r.path, r.before.oid if r.before else None,
                     r.after.oid if r.after else None) for r in codec.iter_changes(cas, base_oid, next_oid)]
        assert observed == expected
    cas.work.clear()
    assert list(codec.iter_changes(cas, root, root)) == []
    assert not cas.work
    assert codec.write_tree(cas, new, previous=changed) == changed
    assert cas.work == {"resolves": 1, "reads": 1}


def test_sparse_publication_and_metadata_update_work(cas):
    leaf = tree.File(0o644, 1, _digest(b"x"))
    names = [f"path-{number:04}".encode() for number in range(5000)]
    before = tree.directory(0o755, ((name, leaf) for name in names))
    old_root = codec.write_tree(cas, before)
    old_meta = codec.write_restore_meta(
        cas, [((), 1, None), *(((name,), 2, None) for name in names)], previous=None,
    )
    cas.work.clear()
    after = tree.replace(before, names[12], tree.File(0o600, 1, _digest(b"y")))
    new_root = codec.write_tree(cas, after, previous=old_root)
    assert new_root != old_root
    assert 1 <= cas.work["publications"] <= 12
    assert 1 <= cas.work["reads"] <= 12
    cas.work.clear()
    new_meta = codec.write_restore_meta(cas, [((names[12],), 3, None)], previous=old_meta,
                                       removed=[(names[14],)])
    assert new_meta != old_meta
    assert 1 <= cas.work["publications"] <= 12
    assert 1 <= cas.work["reads"] <= 60
    rebuilt = codec.write_restore_meta(
        cas,
        [((), 1, None), *(((name,), 3 if name == names[12] else 2, None)
                         for name in names if name != names[14])],
        previous=None,
    )
    assert new_meta == rebuilt


def _policy(**changes):
    return codec.CheckoutPolicy(**{
        "allow_symlinks": False, "allow_internal_hardlinks": False,
        "mode_mask": 0o777, "max_entries": 30, "max_file_bytes": 1024,
        "max_total_bytes": 1024, **changes,
    })


def test_private_checkout_modes_links_and_metadata(cas, tmp_path):
    blob = _store(cas, b"bytes")
    hardlinked = tree.File(0o4640, 5, bytes.fromhex(blob), hardlinked=True)
    leaf = tree.File(0o644, 5, bytes.fromhex(blob))
    root = codec.write_tree(cas, tree.directory(0o1750, [
        (b"dir", tree.directory(0o700, [(b"alias", hardlinked)])),
        (b"leaf", leaf), (b"manual", hardlinked), (b"raw\xff", tree.Symlink(0o777, b"../manual")),
        (b"empty", tree.directory(0o000, [])),
    ]))
    rows = [((), 12345, None), ((b"dir",), 12346, None), ((b"leaf",), 12347, None),
            ((b"manual",), 12348, None), ((b"dir", b"alias"), 12348, (b"manual",)),
            ((b"raw\xff",), 12349, None), ((b"empty",), 12350, None)]
    meta = codec.write_restore_meta(cas, reversed(rows), previous=None)
    assert meta == codec.write_restore_meta(cas, rows, previous=None)
    assert len(list(codec.metadata_edges(cas, meta))) == 1
    destination = tmp_path / "output"
    policy = _policy(allow_symlinks=True, allow_internal_hardlinks=True)
    codec.checkout_tree(cas, root, destination, metadata=meta, policy=policy)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o750
    assert destination.stat().st_mtime_ns == 12345
    assert (destination / "empty").stat().st_mtime_ns == 12350
    assert stat.S_IMODE((destination / "empty").stat().st_mode) == 0
    assert os.readlink(os.fsencode(destination) + b"/raw\xff") == b"../manual"
    assert (destination / os.fsdecode(b"raw\xff")).lstat().st_mtime_ns == 12349
    source, alias = destination / "manual", destination / "dir" / "alias"
    assert source.stat().st_ino == alias.stat().st_ino
    assert source.stat().st_nlink == 2
    assert alias.stat().st_mtime_ns == 12348
    assert stat.S_IMODE(source.stat().st_mode) == 0o640
    source.write_bytes(b"local")
    assert alias.read_bytes() == b"local"
    assert (destination / "leaf").read_bytes() == cas.objects[blob] == b"bytes"
    with pytest.raises(FileExistsError):
        codec.checkout_tree(cas, root, destination, metadata=meta, policy=policy)
    assert source.read_bytes() == b"local"


@pytest.mark.parametrize("defect", ["missing", "wrong-kind", "duplicate", "prefix", "split",
                                    "terminal", "overlong", "bad-order", "mode", "wrong-size"])
def test_malformed_tree_fails_closed(cas, tmp_path, defect):
    file_oid = _store(cas, _raw(b"file", _uint(0o644), _uint(1), _digest(b"x"), b"\0"))
    pointer = bytes.fromhex(file_oid)
    row = _field(b"x") + pointer
    good = _raw(b"bucket", row)
    if defect in {"missing", "wrong-kind", "mode", "wrong-size"}:
        if defect == "missing":
            pointer = b"\0" * 32
        elif defect == "wrong-kind":
            pointer = bytes.fromhex(_store(cas, good))
        elif defect == "mode":
            pointer = bytes.fromhex(_store(
                cas, _raw(b"file", _uint(0o10000), _uint(1), _digest(b"x"), b"\0")
            ))
        else:
            _store(cas, b"x")
            pointer = bytes.fromhex(_store(
                cas, _raw(b"file", _uint(0o644), _uint(2), _digest(b"x"), b"\0")
            ))
        page = _raw(b"bucket", _field(b"x") + pointer)
    elif defect == "duplicate":
        page = _raw(b"bucket", row, row)
    elif defect in {"prefix", "split", "terminal"}:
        child = bytes.fromhex(_store(cas, good))
        slot = b"y" if defect == "prefix" else b"x"
        page = _raw(b"branch", row if defect == "terminal" else b"", slot + child)
    elif defect == "overlong":
        page = _raw(b"bucket", _field(b"x" * 256) + pointer)
    else:
        page = _raw(b"bucket", _field(b"z") + pointer, row)
    oid = _store(cas, _dir(_store(cas, page)))
    with pytest.raises(ObjectIntegrityError):
        codec.checkout_tree(cas, oid, tmp_path / "checkout", metadata=None, policy=_policy())
    assert list(tmp_path.iterdir()) == []


def test_restore_limits_count_logical_aliases_and_refuse_external_group(cas, tmp_path):
    blob = _store(cas, b"bytes")
    file = tree.File(0o644, 5, bytes.fromhex(blob), hardlinked=True)
    subtree = tree.directory(0o755, [(b"file", file)])
    oid = codec.write_tree(cas, tree.directory(0o755, [(b"a", subtree), (b"b", subtree)]))
    for policy, reason in [
        (_policy(allow_internal_hardlinks=True, max_total_bytes=9), "byte limit"),
        (_policy(allow_internal_hardlinks=True, max_entries=3), "entry limit"),
        (_policy(allow_internal_hardlinks=True), "no internal group"),
        (_policy(), "disallows hardlinks"),
    ]:
        with pytest.raises(ObjectIntegrityError, match=reason):
            codec.checkout_tree(cas, oid, tmp_path / "checkout", metadata=None, policy=policy)
    assert list(tmp_path.iterdir()) == []


def test_checkout_rolls_back_failed_finalization(cas, tmp_path, monkeypatch):
    oid = codec.write_tree(cas, tree.directory(
        0o000, [(b"private", tree.directory(0o000, []))]
    ))
    original = codec._set_metadata

    def fail_after_mode(fd, mode, mtime):
        original(fd, mode, mtime)
        if os.fstat(fd).st_ino == stage_inode[0]:
            raise OSError("injected root sync failure")

    stage_inode = [None]
    open_stage = codec._materialize

    def remember_stage(lease, fd, *args):
        stage_inode[0] = os.fstat(fd).st_ino
        return open_stage(lease, fd, *args)

    monkeypatch.setattr(codec, "_materialize", remember_stage)
    monkeypatch.setattr(codec, "_set_metadata", fail_after_mode)
    with pytest.raises(OSError, match="root sync failure"):
        codec.checkout_tree(cas, oid, tmp_path / "checkout", metadata=None, policy=_policy())
    assert list(tmp_path.iterdir()) == []


def test_reload_verified_e5_objects_in_fresh_process(cas, tmp_path):
    # E5 supplies verified storage. On the prerequisite tree only, the reader
    # bridges the future resolve method to E5's lease and verified read.
    home = tmp_path / "private"
    home.mkdir(mode=0o700)
    namespace = "a" * 64
    store = open_workspace_objects(home, namespace, enabled=True)
    file = tree.File(0o644, 1, _digest(b"x"))
    with store.lease() as lease:
        _store(lease, b"x")
        oid = codec.write_tree(lease, tree.directory(0o755, [(b"file", file)]))
    process = subprocess.run([sys.executable, "-c", """
import os
import sys
from pathlib import Path
sys.path.append("tests")
from test_workspace_tree_codec import ObjectRef, codec, open_workspace_objects, registry_stub
if (registry := registry_stub()) is not None:
    sys.modules[registry.__name__] = registry
home, namespace, oid = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
with open_workspace_objects(home, namespace, enabled=True).lease() as lease:
    if not hasattr(lease, "resolve"):
        def resolve(digest, *, max_bytes):
            size = os.stat(digest, dir_fd=lease._check(), follow_symlinks=False).st_size
            assert size <= max_bytes
            return ObjectRef(digest, size)
        lease.resolve = resolve
    leaf = codec.lookup_ref(lease, oid, (b"file",))
    blob, = codec.child_refs(lease, leaf)
    assert blob.kind == "blob"
    assert lease.read(ObjectRef(blob.oid, 1)) == b"x"
    print(leaf.oid)
""", str(home), namespace, oid], capture_output=True, text=True, check=True, timeout=30)
    assert process.stdout.strip() == file.digest.hex()


def test_radix_and_directory_depths_are_independent(cas):
    leaf = tree.File(0o644, 1, _digest(b"x"))
    files = [(b"p" * 254 + bytes((i,)), leaf) for i in range(128, 145)]
    node = tree.directory(0o755, files)
    for _ in range(256):
        node = tree.directory(0o755, [(b"d", node)])
    # A child of that directory would occupy component 257.
    with pytest.raises(ObjectIntegrityError, match="directory depth"):
        codec.write_tree(cas, node)
    node = tree.lookup(node, (b"d",))
    oid = codec.write_tree(cas, node)
    path = (b"d",) * 255 + (files[-1][0],)
    assert codec.lookup_ref(cas, oid, path).oid == leaf.digest.hex()
    with pytest.raises(ValueError, match="relative"):
        codec.lookup_ref(cas, oid, (*path, b"d"))


def test_metadata_edge_mismatch_and_bad_alias_are_refused(cas, tmp_path):
    data = _store(cas, b"bytes")
    leaf = tree.File(0o644, 5, bytes.fromhex(data), hardlinked=True)
    oid = codec.write_tree(cas, tree.directory(0o755, [(b"a", leaf), (b"b", leaf)]))
    valid = codec.write_restore_meta(cas, [((), 0, None)], previous=None)
    tampered = cas.objects[valid.oid].replace(b'"edges":[["meta","', b'"edges":[["blob","', 1)
    forged = type(valid)("meta", _store(cas, tampered))
    with pytest.raises(ObjectIntegrityError, match="edges disagree"):
        list(codec.metadata_edges(cas, forged))
    meta = codec.write_restore_meta(cas, [
        ((b"a",), 1, (b"a",)), ((b"b",), 2, (b"a",)),
    ], previous=valid)
    with pytest.raises(ObjectIntegrityError, match="disagrees with its leader"):
        codec.checkout_tree(cas, oid, tmp_path / "checkout", metadata=meta,
                            policy=_policy(allow_internal_hardlinks=True))
    assert not list(tmp_path.iterdir())


def test_bucket_split_does_not_republish_unchanged_entries(cas):
    # Distinct descriptors make each unnecessary publication visible.
    files = [(bytes((65 + number,)), tree.File(0o644, 1, _digest(bytes((number,)))))
             for number in range(17)]
    before = tree.directory(0o755, files[:16])
    old = codec.write_tree(cas, before)
    after = tree.replace(before, *files[-1])
    cas.work.clear()
    new = codec.write_tree(cas, after, previous=old)
    # One branch, 17 buckets, one new entry descriptor and one root.
    assert cas.work["publications"] == 20
    assert [change.path for change in codec.iter_changes(cas, old, new)] == [(files[-1][0],)]
