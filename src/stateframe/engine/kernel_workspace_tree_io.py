"""Persist workspace identities and traverse immutable objects under a store lease.

Callers authorize blob retention before publication, keep committed roots pinned,
and own the destination lock for checkout. Tree objects contain references only.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import stat
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from kernel_workspace_objects import ObjectIntegrityError, ObjectRef, _directory
from kernel_workspace_tree import Directory, Entry, _BUCKET_SIZE, _DOMAIN, _Page, _join, validate_name

if TYPE_CHECKING:
    from kernel_workspace_objects import ObjectLease
    from kernel_workspace_refs import Ref


Digest = str
Rel = tuple[bytes, ...]
_RefKind = Literal["tree", "node", "blob", "meta"]
_MAX_DEPTH = 256
_MAX_NAME = 255
_MAX_SIZE = (1 << 63) - 1
_HEX_OID = re.compile(r"[0-9a-f]{64}")
_NODE_BYTES = 64 << 10
_META_BYTES = 256 << 10
_META_ROOT = "stateframe-restore-meta/1"
_META_DIR = "stateframe-restore-dir/1"
_META_PAGE = "stateframe-restore-page/1"
_SPECIAL_TYPES = {stat.S_IFBLK, stat.S_IFCHR, stat.S_IFIFO, stat.S_IFSOCK}


def _ref(kind: _RefKind, oid: bytes) -> Ref:
    # The registry also uses the codec to trace committed references.
    from kernel_workspace_refs import Ref

    return Ref(kind, oid.hex())


def _oid(value: object) -> bytes:
    if not isinstance(value, str) or _HEX_OID.fullmatch(value) is None:
        raise ObjectIntegrityError("invalid workspace object ID")
    return bytes.fromhex(value)


def _path(path: Rel) -> Rel:
    if not isinstance(path, tuple) or len(path) > _MAX_DEPTH:
        raise ValueError("invalid relative workspace path")
    for name in path:
        validate_name(name)
    return path


def _read(lease: ObjectLease, oid: bytes, bound: int) -> bytes:
    try:
        return lease.read(lease.resolve(oid.hex(), max_bytes=bound))
    except FileNotFoundError as error:
        raise ObjectIntegrityError("missing workspace tree or metadata object") from error


def _field(data: bytes, start: int) -> tuple[bytes, int]:
    if start + 4 > len(data):
        raise ObjectIntegrityError("truncated workspace field length")
    size = int.from_bytes(data[start:start + 4], "big")
    end = start + 4 + size
    if end > len(data):
        raise ObjectIntegrityError("truncated workspace field")
    return data[start + 4:end], end


def _parts(data: bytes) -> tuple[bytes, list[bytes]]:
    if not data.startswith(_DOMAIN):
        raise ObjectIntegrityError("invalid workspace tree domain")
    tag, offset = _field(data, len(_DOMAIN))
    fields = []
    while offset < len(data):
        if len(fields) == 257:
            raise ObjectIntegrityError("too many workspace fields")
        value, offset = _field(data, offset)
        fields.append(value)
    return tag, fields


def _uint(value: bytes, maximum: int = (1 << 64) - 1) -> int:
    if len(value) != 8 or (number := int.from_bytes(value, "big")) > maximum:
        raise ObjectIntegrityError("invalid workspace integer")
    return number


def _pointer(value: bytes) -> bytes:
    if len(value) != 32:
        raise ObjectIntegrityError("invalid workspace tree pointer")
    return value


def _name(value: bytes) -> bytes:
    try:
        validate_name(value)
    except ValueError as error:
        raise ObjectIntegrityError("invalid stored workspace name") from error
    return value


_Row = tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class _Radix:
    rows: tuple[_Row, ...] = ()
    terminal: _Row | None = None
    edges: tuple[tuple[int, bytes], ...] = ()

    def __post_init__(self) -> None:
        if self.rows:
            keys = [name for name, _ in self.rows]
            if len(keys) > _BUCKET_SIZE or keys != sorted(set(keys)):
                raise ObjectIntegrityError("invalid workspace bucket order or count")
        else:
            keys = [slot for slot, _ in self.edges]
            if not keys or keys != sorted(set(keys)):
                raise ObjectIntegrityError("invalid workspace branch order or count")

    def validate_prefix(self, prefix: bytes) -> None:
        if self.rows:
            if any(not name.startswith(prefix) for name, _ in self.rows):
                raise ObjectIntegrityError("workspace bucket is in the wrong radix slot")
        elif len(prefix) >= _MAX_NAME or (self.terminal and self.terminal[0] != prefix):
            raise ObjectIntegrityError("invalid workspace branch prefix")


def _row(data: bytes) -> _Row:
    name, end = _field(data, 0)
    return _name(name), _pointer(data[end:])


def _decode_page(tag: bytes, fields: list[bytes]) -> _Radix:
    if tag == b"bucket" and fields:
        return _Radix(rows=tuple(map(_row, fields)))
    if tag == b"branch" and len(fields) >= 2:
        edges = []
        for edge in fields[1:]:
            if len(edge) != 33:
                raise ObjectIntegrityError("invalid workspace branch edge")
            edges.append((edge[0], edge[1:]))
        return _Radix(terminal=_row(fields[0]) if fields[0] else None, edges=tuple(edges))
    raise ObjectIntegrityError("invalid workspace page tag or arity")


@dataclass(frozen=True, slots=True)
class TreeEntry:
    """Decoded entry descriptor. File payloads stay in the object store."""

    kind: Literal["dir", "file", "symlink", "special"]
    mode: int
    size: int = 0
    sha256: bytes | None = None
    hardlinked: bool = False
    target: bytes | None = None
    file_type: int = 0
    device: int = 0
    _page: bytes | None = field(default=None, repr=False)


def _decode_entry(tag: bytes, fields: list[bytes]) -> TreeEntry:
    if tag == b"dir" and len(fields) == 2:
        return TreeEntry("dir", _uint(fields[0], 0o7777),
                         _page=_pointer(fields[1]) if fields[1] else None)
    if tag == b"file" and len(fields) == 4 and fields[3] in (b"\0", b"\1"):
        return TreeEntry("file", _uint(fields[0], 0o7777), _uint(fields[1], _MAX_SIZE),
                         _pointer(fields[2]), fields[3] == b"\1")
    if tag == b"symlink" and len(fields) == 2 and fields[1] and b"\0" not in fields[1]:
        return TreeEntry("symlink", _uint(fields[0], 0o7777), target=fields[1])
    if tag == b"special" and len(fields) == 4 and (kind := _uint(fields[1])) in _SPECIAL_TYPES:
        return TreeEntry("special", _uint(fields[0], 0o7777), _uint(fields[3], _MAX_SIZE),
                         file_type=kind, device=_uint(fields[2]))
    raise ObjectIntegrityError("invalid workspace entry tag or fields")


class _Reader:
    def __init__(self, lease: ObjectLease):
        self.lease = lease

    def node(self, oid: bytes) -> TreeEntry | _Radix:
        tag, fields = _parts(_read(self.lease, oid, _NODE_BYTES))
        if tag in (b"branch", b"bucket"):
            return _decode_page(tag, fields)
        return _decode_entry(tag, fields)

    def entry(self, oid: bytes, *, root: bool = False) -> TreeEntry:
        node = self.node(oid)
        if isinstance(node, _Radix) or (root and node.kind != "dir"):
            raise ObjectIntegrityError("workspace reference has the wrong node type")
        return node

    def page(self, oid: bytes) -> _Radix:
        page = self.node(oid)
        if not isinstance(page, _Radix):
            raise ObjectIntegrityError("workspace radix edge is not a page")
        return page


_ReadPage = Callable[[bytes], _Radix]


def _page_at(read: _ReadPage, oid: bytes, prefix: bytes) -> _Radix:
    page = read(oid)
    page.validate_prefix(prefix)
    return page


def _rows(read: _ReadPage, oid: bytes | None, prefix: bytes = b"") -> Iterator[_Row]:
    if oid is not None:
        yield from _page_rows(read, _page_at(read, oid, prefix), prefix)


def _page_rows(read: _ReadPage, page: _Radix, prefix: bytes) -> Iterator[_Row]:
    if page.rows:
        yield from page.rows
        return
    count = 0
    if page.terminal is not None:
        count += 1
        yield page.terminal
    for slot, child in page.edges:
        for row in _rows(read, child, prefix + bytes((slot,))):
            count += 1
            yield row
    if count <= _BUCKET_SIZE:
        raise ObjectIntegrityError("noncanonical workspace branch split")


def _find(read: _ReadPage, oid: bytes | None, name: bytes, prefix: bytes = b"") -> bytes | None:
    while oid is not None:
        page = _page_at(read, oid, prefix)
        if page.rows:
            return dict(page.rows).get(name)
        if prefix == name:
            return page.terminal[1] if page.terminal else None
        slot = name[len(prefix)]
        oid = dict(page.edges).get(slot)
        prefix += bytes((slot,))
    return None


def _page_changes(
    read: _ReadPage, old: bytes | None, new: bytes | None, prefix: bytes = b"",
) -> Iterator[tuple[bytes, bytes | None, bytes | None]]:
    if old == new:
        return
    before = _page_at(read, old, prefix) if old else None
    after = _page_at(read, new, prefix) if new else None
    if before and after and before.edges and after.edges:
        yield from _join((before.terminal,) if before.terminal else (),
                         (after.terminal,) if after.terminal else ())
        for slot, left, right in _join(before.edges, after.edges):
            yield from _page_changes(read, left, right, prefix + bytes((slot,)))
    else:
        yield from _join(_page_rows(read, before, prefix) if before else (),
                         _page_rows(read, after, prefix) if after else ())


@dataclass(frozen=True, slots=True)
class ChangeRef:
    path: Rel
    before: Ref | None
    after: Ref | None


def _entry_ref(oid: bytes, entry: TreeEntry) -> Ref:
    return _ref("tree" if entry.kind == "dir" else "node", oid)


def iter_changes(lease: ObjectLease, old: Digest, new: Digest) -> Iterator[ChangeRef]:
    """Yield changes in raw path order, visiting only unequal object edges."""
    reader = _Reader(lease)
    stack = [iter((((), _oid(old), _oid(new)),))]
    while stack:
        row = next(stack[-1], None)
        if row is None:
            stack.pop()
            continue
        path, left, right = row
        if left == right:
            continue
        if len(path) > _MAX_DEPTH:
            raise ObjectIntegrityError("workspace directory depth exceeds limit")
        before = reader.entry(left, root=not path) if left is not None else None
        after = reader.entry(right, root=not path) if right is not None else None
        if (before is None or after is None or before.kind != "dir" or after.kind != "dir"
                or before.mode != after.mode):
            yield ChangeRef(path, _entry_ref(left, before) if before else None,
                            _entry_ref(right, after) if after else None)
        pages = _page_changes(reader.page, before._page if before else None,
                              after._page if after else None)
        stack.append(_prepend(path, pages))


def _prepend(
    path: Rel, rows: Iterable[tuple[bytes, bytes | None, bytes | None]],
) -> Iterator[tuple[Rel, bytes | None, bytes | None]]:
    for name, before, after in rows:
        yield (*path, name), before, after


def lookup_ref(lease: ObjectLease, tree: Digest, path: Rel) -> Ref | None:
    """Resolve a raw path without following a symlink or loading siblings."""
    _path(path)
    reader = _Reader(lease)
    oid = _oid(tree)
    entry = reader.entry(oid, root=True)
    for name in path:
        child = _find(reader.page, entry._page, name)
        if child is None:
            return None
        oid = child
        entry = reader.entry(oid)
    return _entry_ref(oid, entry)


def read_entry(lease: ObjectLease, obj: Ref) -> TreeEntry:
    """Read an entry returned by lookup or diff and validate its descriptor."""
    if obj.kind not in ("tree", "node"):
        raise ObjectIntegrityError("invalid workspace entry reference kind")
    return _Reader(lease).entry(_oid(obj.oid), root=obj.kind == "tree")


def _entries(reader: _Reader, tree: Digest) -> Iterator[tuple[Rel, bytes, TreeEntry]]:
    stack = [((), iter(((b"", _oid(tree)),)))]
    while stack:
        parent, rows = stack[-1]
        row = next(rows, None)
        if row is None:
            stack.pop()
            continue
        name, oid = row
        path = (*parent, name) if name else ()
        if len(path) > _MAX_DEPTH:
            raise ObjectIntegrityError("workspace directory depth exceeds limit")
        entry = reader.entry(oid, root=not path)
        yield path, oid, entry
        if entry._page is not None:
            stack.append((path, _rows(reader.page, entry._page)))


def iter_entries(lease: ObjectLease, tree: Digest) -> Iterator[tuple[Rel, Ref, TreeEntry]]:
    """Walk logical entries, including the root, without loading any blobs."""
    for path, oid, entry in _entries(_Reader(lease), tree):
        yield path, _entry_ref(oid, entry), entry


def child_refs(lease: ObjectLease, obj: Ref) -> Iterator[Ref]:
    """Return immediate typed CAS edges for GC."""
    if obj.kind not in ("tree", "node"):
        raise ObjectIntegrityError("invalid workspace tree reference kind")
    reader = _Reader(lease)
    oid = _oid(obj.oid)
    node = reader.entry(oid, root=True) if obj.kind == "tree" else reader.node(oid)
    if isinstance(node, _Radix):
        if node.terminal is not None:
            yield _ref("node", node.terminal[1])
        for _, child in (*node.rows, *node.edges):
            yield _ref("node", child)
    elif node._page is not None:
        yield _ref("node", node._page)
    elif node.sha256 is not None:
        yield _ref("blob", node.sha256)


class _Publication:
    def __init__(self, lease: ObjectLease):
        self.lease = lease
        self.pending: dict[bytes, bytes] = {}

    def publish(self, order: Iterable[bytes] | None = None) -> None:
        if self.pending:
            self.lease.publish_many(
                (ObjectRef(oid.hex(), len(data)), partial(BytesIO, data))
                for oid in (self.pending if order is None else order)
                for data in (self.pending[oid],)
            )


_PreviousPage = bytes | _Radix | None
_TreeTask = tuple[Entry | _Page, _PreviousPage, bytes, int]


class _TreePublication(_Publication):
    def __init__(self, lease: ObjectLease):
        super().__init__(lease)
        self.reader = _Reader(lease)

    def _page_children(self, page: _Page, previous: _PreviousPage,
                       prefix: bytes, depth: int) -> Iterator[_TreeTask]:
        old = _page_at(self.reader.page, previous, prefix) if isinstance(previous, bytes) else previous
        old_rows = dict(old.rows) if old else {}
        if page.rows:
            for name, entry in page.rows:
                if old and old.edges:
                    assert isinstance(previous, bytes)
                    previous_entry = _find(self.reader.page, previous, name, prefix)
                else:
                    previous_entry = old_rows.get(name)
                yield entry, previous_entry, b"", depth + 1
        else:
            if page.terminal is not None:
                name, entry = page.terminal
                old_terminal = old.terminal[1] if old and old.terminal else old_rows.get(name)
                yield entry, old_terminal, b"", depth + 1
            old_edges: dict[int, _PreviousPage] = dict(old.edges) if old else {}
            if old_rows:
                # Splitting a bucket changes page identities. Its same-name
                # entry descriptors are still reusable in the new slots.
                groups: dict[int, list[_Row]] = {}
                for name, oid in old_rows.items():
                    if len(name) > len(prefix):
                        groups.setdefault(name[len(prefix)], []).append((name, oid))
                old_edges.update((slot, _Radix(rows=tuple(rows))) for slot, rows in groups.items())
            for slot, child in page.edges:
                yield child, old_edges.get(slot), prefix + bytes((slot,)), depth

    def stage(self, root: Directory, previous: bytes | None) -> None:
        if not isinstance(root, Directory):
            raise ValueError("workspace publication requires an E2 directory")
        stack: list[_TreeTask | tuple[bytes, bytes]] = [(root, previous, b"", 0)]
        while stack:
            task = stack.pop()
            if len(task) == 2:
                oid, data = task
                self.pending[oid] = data
                continue
            node, old, prefix, depth = task
            if depth > _MAX_DEPTH:
                raise ObjectIntegrityError("workspace directory depth exceeds limit")
            if node.digest == old or node.digest in self.pending:
                continue
            data = node._preimage()
            if len(data) > _NODE_BYTES:
                raise ObjectIntegrityError("workspace node exceeds byte limit")
            stack.append((node.digest, data))
            if isinstance(node, _Page):
                stack.extend(reversed(tuple(self._page_children(node, old, prefix, depth))))
            else:
                entry = _decode_entry(*_parts(data))
                if isinstance(node, Directory) and entry._page is not None:
                    assert old is None or isinstance(old, bytes)
                    old_page = self.reader.entry(old)._page if old is not None else None
                    stack.append((node._page, old_page, b"", depth))


def write_tree(lease: ObjectLease, tree: Directory, *, previous: Digest | None = None) -> Digest:
    """Publish exact E2 preimages. The caller has already published admitted blobs."""
    writer = _TreePublication(lease)
    old = _oid(previous) if previous is not None else None
    if old is not None:
        writer.reader.entry(old, root=True)
    writer.stage(tree, old)
    writer.publish()
    return tree.digest.hex()


# Restore metadata uses its own radix and never changes a semantic tree OID.
_MetaValue = tuple[int, Rel | None]


def _json(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _meta_envelope(schema: str, payload: object, edges: Iterable[bytes]) -> bytes:
    return _json({"schema": schema, "payload": payload,
                  "edges": [["meta", edge.hex()] for edge in edges]})


def _array(value: object, size: int) -> list[object]:
    if not isinstance(value, list) or len(value) != size:
        raise ObjectIntegrityError("invalid restore metadata array")
    return value


def _meta_name(value: object) -> bytes:
    if not isinstance(value, str) or len(value) > 2 * _MAX_NAME:
        raise ObjectIntegrityError("invalid restore metadata name")
    try:
        name = bytes.fromhex(value)
    except ValueError as error:
        raise ObjectIntegrityError("invalid restore metadata name") from error
    if name.hex() != value:
        raise ObjectIntegrityError("noncanonical restore metadata name")
    return _name(name)


def _meta_row(value: object) -> _Row:
    name, oid = _array(value, 2)
    return _meta_name(name), _oid(oid)


def _meta_value(value: object) -> _MetaValue:
    mtime, leader = _array(value, 2)
    if type(mtime) is not int or not -_MAX_SIZE - 1 <= mtime <= _MAX_SIZE:
        raise ObjectIntegrityError("invalid restore metadata timestamp")
    if leader is None:
        return mtime, None
    if not isinstance(leader, list) or not 1 <= len(leader) <= _MAX_DEPTH:
        raise ObjectIntegrityError("invalid restore metadata hardlink leader")
    return mtime, tuple(map(_meta_name, leader))


def _meta_value_json(value: _MetaValue | None) -> object:
    if value is None:
        return None
    mtime, leader = value
    return [mtime, [name.hex() for name in leader] if leader is not None else None]


@dataclass(frozen=True, slots=True)
class _MetaDir:
    value: _MetaValue | None
    page: bytes | None


def _decode_meta_page(payload: object) -> _Radix:
    if not isinstance(payload, list) or not payload:
        raise ObjectIntegrityError("invalid restore metadata page")
    if payload[0] == "bucket":
        _, rows = _array(payload, 2)
        if not isinstance(rows, list) or not 1 <= len(rows) <= _BUCKET_SIZE:
            raise ObjectIntegrityError("invalid restore metadata bucket count")
        return _Radix(rows=tuple(map(_meta_row, rows)))
    if payload[0] != "branch":
        raise ObjectIntegrityError("invalid restore metadata page tag")
    _, terminal, edges = _array(payload, 3)
    if not isinstance(edges, list) or not 1 <= len(edges) <= 256:
        raise ObjectIntegrityError("invalid restore metadata branch count")
    decoded = []
    for edge in edges:
        slot, oid = _array(edge, 2)
        if not isinstance(slot, str) or re.fullmatch(r"[0-9a-f]{2}", slot) is None:
            raise ObjectIntegrityError("invalid restore metadata radix slot")
        decoded.append((int(slot, 16), _oid(oid)))
    return _Radix(terminal=_meta_row(terminal) if terminal is not None else None,
                  edges=tuple(decoded))


def _radix_edges(page: _Radix) -> Iterator[bytes]:
    if page.terminal:
        yield page.terminal[1]
    for _, oid in (*page.rows, *page.edges):
        yield oid


def _meta_node(lease: ObjectLease, oid: bytes) -> tuple[bytes | _MetaDir | _Radix, tuple[bytes, ...]]:
    data = _read(lease, oid, _META_BYTES)
    try:
        envelope = json.loads(data)
        if (not isinstance(envelope, dict) or set(envelope) != {"schema", "payload", "edges"}
                or data != _json(envelope)):
            raise ObjectIntegrityError("noncanonical restore metadata envelope")
    except (ValueError, RecursionError) as error:
        raise ObjectIntegrityError("invalid restore metadata encoding") from error
    schema, payload = envelope["schema"], envelope["payload"]
    if schema == _META_ROOT:
        value = _oid(payload)
        children = (value,)
    elif schema == _META_DIR:
        row, page = _array(payload, 2)
        value = _MetaDir(_meta_value(row) if row is not None else None,
                         _oid(page) if page is not None else None)
        if value.value is None and value.page is None:
            raise ObjectIntegrityError("empty restore metadata directory")
        children = (value.page,) if value.page is not None else ()
    elif schema == _META_PAGE:
        value = _decode_meta_page(payload)
        children = tuple(_radix_edges(value))
    else:
        raise ObjectIntegrityError("unknown restore metadata schema")
    if envelope["edges"] != [["meta", child.hex()] for child in children]:
        raise ObjectIntegrityError("restore metadata edges disagree with payload")
    return value, children


def metadata_edges(lease: ObjectLease, obj: Ref) -> Iterator[Ref]:
    """Return immediate edges after validating the metadata envelope."""
    if obj.kind != "meta":
        raise ObjectIntegrityError("invalid restore metadata reference kind")
    _, edges = _meta_node(lease, _oid(obj.oid))
    for edge in edges:
        yield _ref("meta", edge)


class _MetaReader:
    def __init__(self, lease: ObjectLease):
        self.lease = lease

    def root(self, ref: Ref) -> bytes:
        if ref.kind != "meta":
            raise ObjectIntegrityError("invalid restore metadata root kind")
        node, _ = _meta_node(self.lease, _oid(ref.oid))
        if not isinstance(node, bytes):
            raise ObjectIntegrityError("workspace metadata requires a root envelope")
        return node

    def directory(self, oid: bytes) -> _MetaDir:
        node, _ = _meta_node(self.lease, oid)
        if not isinstance(node, _MetaDir):
            raise ObjectIntegrityError("workspace metadata edge is not a directory")
        return node

    def page(self, oid: bytes) -> _Radix:
        node, _ = _meta_node(self.lease, oid)
        if not isinstance(node, _Radix):
            raise ObjectIntegrityError("workspace metadata edge is not a page")
        return node


@dataclass(slots=True)
class _MetaEdit:
    remove: bool = False
    value: _MetaValue | None = None
    children: dict[bytes, _MetaEdit] = field(default_factory=dict)

    def at(self, path: Rel) -> _MetaEdit:
        edit = self
        for name in _path(path):
            if name not in edit.children:
                edit.children[name] = _MetaEdit()
            edit = edit.children[name]
        return edit


class _MetaPublication(_Publication):
    def __init__(self, lease: ObjectLease):
        super().__init__(lease)
        self.reader = _MetaReader(lease)
        self.links: dict[bytes, tuple[bytes, ...]] = {}
        self.pages: dict[bytes, _Radix] = {}

    def stage(self, schema: str, payload: object, edges: Iterable[bytes]) -> bytes:
        children = tuple(edges)
        data = _meta_envelope(schema, payload, children)
        oid = sha256(data).digest()
        self.pending[oid] = data
        self.links[oid] = children
        return oid

    def read_page(self, oid: bytes) -> _Radix:
        page = self.pages.get(oid)
        return page if page is not None else self.reader.page(oid)

    def store_page(self, page: _Radix) -> bytes:
        if page.rows:
            payload = ["bucket", [[name.hex(), oid.hex()] for name, oid in page.rows]]
        else:
            terminal = [page.terminal[0].hex(), page.terminal[1].hex()] if page.terminal else None
            payload = ["branch", terminal, [[f"{slot:02x}", oid.hex()] for slot, oid in page.edges]]
        oid = self.stage(_META_PAGE, payload, _radix_edges(page))
        self.pages[oid] = page
        return oid

    def build(self, rows: tuple[_Row, ...], depth: int) -> bytes | None:
        if not rows:
            return None
        if len(rows) <= _BUCKET_SIZE:
            return self.store_page(_Radix(rows=rows))
        terminal = rows[0] if len(rows[0][0]) == depth else None
        groups: dict[int, list[_Row]] = {}
        for row in rows[int(terminal is not None):]:
            groups.setdefault(row[0][depth], []).append(row)
        edges = []
        for slot, group in groups.items():
            child = self.build(tuple(group), depth + 1)
            assert child is not None
            edges.append((slot, child))
        return self.store_page(_Radix(terminal=terminal, edges=tuple(edges)))

    def update_page(self, oid: bytes | None, prefix: bytes,
                    edits: dict[bytes, bytes | None]) -> bytes | None:
        if not edits:
            return oid
        page = _page_at(self.read_page, oid, prefix) if oid is not None else None
        if page is None or page.rows:
            rows = dict(page.rows) if page is not None else {}
            for name, child in edits.items():
                if child is None:
                    rows.pop(name, None)
                else:
                    rows[name] = child
            return self.build(tuple(sorted(rows.items())), len(prefix))

        terminal = page.terminal
        edges = dict(page.edges)
        groups: dict[int, dict[bytes, bytes | None]] = {}
        for name, child in edits.items():
            if name == prefix:
                terminal = (name, child) if child is not None else None
            else:
                groups.setdefault(name[len(prefix)], {})[name] = child
        for slot, group in groups.items():
            child = self.update_page(edges.get(slot), prefix + bytes((slot,)), group)
            if child is None:
                edges.pop(slot, None)
            else:
                edges[slot] = child
        ordered = tuple(sorted(edges.items()))
        # The minimum direct fanout suffices to prove that many wide pages
        # cannot collapse. A deletion near the threshold needs at most 17 rows.
        if (int(terminal is not None) + len(edges) <= _BUCKET_SIZE
                and any(child is None for child in edits.values())):
            rows = [terminal] if terminal else []
            for slot, child in ordered:
                for row in _rows(self.read_page, child, prefix + bytes((slot,))):
                    rows.append(row)
                    if len(rows) > _BUCKET_SIZE:
                        return self.store_page(_Radix(terminal=terminal, edges=ordered))
            return self.build(tuple(rows), len(prefix))
        return self.store_page(_Radix(terminal=terminal, edges=ordered))

    def edit_directory(self, oid: bytes | None, edit: _MetaEdit, *, root: bool = False) -> bytes | None:
        if edit.remove:
            oid = None
        old = self.reader.directory(oid) if oid is not None else _MetaDir(None, None)
        value = old.value if edit.value is None else edit.value
        if root and value is None:
            raise ObjectIntegrityError("workspace restore metadata requires a root row")
        edits = {}
        for name, child_edit in edit.children.items():
            child = _find(self.read_page, old.page, name)
            updated = self.edit_directory(child, child_edit)
            if child != updated:
                edits[name] = updated
        page = self.update_page(old.page, b"", edits)
        if value is None and page is None:
            return None
        if value == old.value and page == old.page:
            return oid
        return self.stage(_META_DIR, [_meta_value_json(value), page.hex() if page else None],
                          () if page is None else (page,))

    def publish_root(self, oid: bytes) -> None:
        reachable = set()
        pending = [oid]
        while pending:
            node = pending.pop()
            if node not in reachable and node in self.links:
                reachable.add(node)
                pending.extend(self.links[node])
        self.publish(oid for oid in self.pending if oid in reachable)


def write_restore_meta(
    lease: ObjectLease, rows: Iterable[tuple[Rel, int, Rel | None]], *,
    previous: Ref | None, removed: Iterable[Rel] = (),
) -> Ref:
    """Publish complete metadata or update affected paths of a committed trie."""
    edits = _MetaEdit()
    for path in removed:
        edits.at(path).remove = True
    for path, mtime, leader in rows:
        if type(mtime) is not int or not -_MAX_SIZE - 1 <= mtime <= _MAX_SIZE:
            raise ValueError("invalid workspace restore mtime")
        if leader is not None and not _path(leader):
            raise ValueError("invalid workspace hardlink leader")
        edit = edits.at(path)
        if edit.value is not None:
            raise ValueError("duplicate workspace restore metadata path")
        edit.value = (mtime, leader)

    writer = _MetaPublication(lease)
    old = writer.reader.root(previous) if previous is not None else None
    new = writer.edit_directory(old, edits, root=True)
    assert new is not None
    if new == old:
        assert previous is not None
        return previous
    oid = writer.stage(_META_ROOT, new.hex(), (new,))
    writer.publish_root(oid)
    return _ref("meta", oid)


def _metadata(lease: ObjectLease, ref: Ref, paths: dict[Rel, TreeEntry]) -> dict[Rel, _MetaValue]:
    reader = _MetaReader(lease)
    stack = [((), iter(((b"", reader.root(ref)),)))]
    result = {}
    while stack:
        parent, rows = stack[-1]
        row = next(rows, None)
        if row is None:
            stack.pop()
            continue
        name, oid = row
        path = (*parent, name) if name else ()
        if path not in paths:
            raise ObjectIntegrityError("restore metadata path is outside the workspace tree")
        node = reader.directory(oid)
        if node.value is not None:
            result[path] = node.value
        if node.page is not None:
            if paths[path].kind != "dir":
                raise ObjectIntegrityError("restore metadata descends through a leaf")
            stack.append((path, _rows(reader.page, node.page)))
    if () not in result:
        raise ObjectIntegrityError("workspace restore metadata requires a root row")
    return result


@dataclass(frozen=True, slots=True)
class CheckoutPolicy:
    allow_symlinks: bool
    allow_internal_hardlinks: bool
    mode_mask: int
    max_entries: int
    max_file_bytes: int
    max_total_bytes: int

    def __post_init__(self) -> None:
        if (type(self.allow_symlinks) is not bool or type(self.allow_internal_hardlinks) is not bool
                or type(self.mode_mask) is not int or not 0 <= self.mode_mask <= 0o7777
                or any(type(limit) is not int or limit < 0 for limit in (
                    self.max_entries, self.max_file_bytes, self.max_total_bytes
                ))):
            raise ValueError("invalid workspace checkout policy")


def _checkout_entries(lease: ObjectLease, tree: Digest, policy: CheckoutPolicy) -> dict[Rel, TreeEntry]:
    entries = {}
    total = 0
    for path, _, entry in _entries(_Reader(lease), tree):
        if path and len(entries) > policy.max_entries:
            raise ObjectIntegrityError("workspace checkout exceeds entry limit")
        if entry.kind == "file":
            total += entry.size
            if entry.size > policy.max_file_bytes or total > policy.max_total_bytes:
                raise ObjectIntegrityError("workspace checkout exceeds byte limit")
            if entry.hardlinked and not policy.allow_internal_hardlinks:
                raise ObjectIntegrityError("workspace checkout disallows hardlinks")
        elif entry.kind == "special" or (entry.kind == "symlink" and not policy.allow_symlinks):
            raise ObjectIntegrityError("workspace checkout disallows entry kind")
        entries[path] = entry
    return entries


def _aliases(entries: dict[Rel, TreeEntry], meta: dict[Rel, _MetaValue]) -> dict[Rel, Rel]:
    aliases = {}
    leaders = set()
    for path, (mtime, leader) in meta.items():
        if leader is None:
            continue
        entry = entries[path]
        source = entries.get(leader)
        source_time, source_leader = meta.get(leader, (None, None))
        if (not entry.hardlinked or source is None or source != entry
                or source_time != mtime or source_leader not in (None, leader)):
            raise ObjectIntegrityError("workspace hardlink group disagrees with its leader")
        if path != leader:
            aliases[path] = leader
            leaders.add(leader)
    for path, entry in entries.items():
        if entry.hardlinked and path not in aliases and path not in leaders:
            raise ObjectIntegrityError("workspace hardlink has no internal group")
    return aliases


_CHECKOUT_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_relative(root: int, path: Rel) -> int:
    directory = os.dup(root)
    try:
        for name in path:
            child = os.open(name, _CHECKOUT_DIR_FLAGS, dir_fd=directory)
            os.close(directory)
            directory = child
    except BaseException:
        os.close(directory)
        raise
    return directory


def _set_metadata(fd: int, mode: int, mtime: int | None) -> None:
    os.fchmod(fd, mode)
    if mtime is not None:
        os.utime(fd, ns=(mtime, mtime))
    os.fsync(fd)


def _restore_file(
    lease: ObjectLease, directory: int, name: bytes, entry: TreeEntry,
    mtime: int | None, policy: CheckoutPolicy,
) -> None:
    assert entry.sha256 is not None
    try:
        ref = lease.resolve(entry.sha256.hex(), max_bytes=policy.max_file_bytes)
        if ref.size != entry.size:
            raise ObjectIntegrityError("workspace blob size differs from file descriptor")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as output:
            lease.copy_verified(ref, output, max_bytes=policy.max_file_bytes)
            output.flush()
            _set_metadata(output.fileno(), entry.mode & policy.mode_mask, mtime)
    except FileNotFoundError as error:
        raise ObjectIntegrityError("missing workspace blob") from error


def _materialize(
    lease: ObjectLease, root: int, entries: dict[Rel, TreeEntry],
    meta: dict[Rel, _MetaValue], aliases: dict[Rel, Rel], policy: CheckoutPolicy,
    directories: list[Rel],
) -> None:
    for path, entry in entries.items():
        if not path or path in aliases:
            continue
        parent, name = _open_relative(root, path[:-1]), path[-1]
        mtime, _ = meta.get(path, (None, None))
        try:
            if entry.kind == "dir":
                os.mkdir(name, 0o700, dir_fd=parent)
                directories.append(path)
                os.chmod(name, 0o700, dir_fd=parent, follow_symlinks=False)
            elif entry.kind == "symlink":
                # The host has admitted this opaque link text. No referent is opened.
                os.symlink(entry.target, name, dir_fd=parent)
                if mtime is not None:
                    os.utime(name, ns=(mtime, mtime), dir_fd=parent, follow_symlinks=False)
            else:
                _restore_file(lease, parent, name, entry, mtime, policy)
        finally:
            os.close(parent)

    for path, leader in aliases.items():
        source = _open_relative(root, leader[:-1])
        try:
            parent = _open_relative(root, path[:-1])
            try:
                os.link(leader[-1], path[-1], src_dir_fd=source, dst_dir_fd=parent,
                        follow_symlinks=False)
            finally:
                os.close(parent)
        finally:
            os.close(source)

    for path in reversed(directories):
        directory = _open_relative(root, path)
        try:
            mtime, _ = meta.get(path, (None, None))
            _set_metadata(directory, entries[path].mode & policy.mode_mask, mtime)
        finally:
            os.close(directory)


def _require_absent(parent: int, name: str) -> None:
    try:
        os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError("workspace checkout destination already exists")


def _discard_stage(parent: int, stage: str, root: int, directories: list[Rel]) -> None:
    # Final mode installation may have made some directories unsearchable.
    os.fchmod(root, 0o700)
    for path in directories[1:]:
        fd = _open_relative(root, path[:-1])
        try:
            os.chmod(path[-1], 0o700, dir_fd=fd, follow_symlinks=False)
        finally:
            os.close(fd)
    shutil.rmtree(stage, dir_fd=parent)


def checkout_tree(
    lease: ObjectLease, tree: Digest, destination: Path, *,
    metadata: Ref | None, policy: CheckoutPolicy,
) -> None:
    """Build a private checkout and publish it under the caller's destination lock.

    A consumer opting into symlinks must authorize their text and apply any
    protected-home exclusions before calling. The writer never dereferences links.
    """
    destination = Path(destination)
    if destination.name in ("", ".."):
        raise ValueError("invalid workspace checkout destination")
    entries = _checkout_entries(lease, tree, policy)
    meta = _metadata(lease, metadata, entries) if metadata is not None else {}
    aliases = _aliases(entries, meta)
    with _directory(destination.parent) as parent:
        _require_absent(parent, destination.name)
        stage = ".checkout-" + secrets.token_hex(16)
        os.mkdir(stage, 0o700, dir_fd=parent)
        try:
            os.chmod(stage, 0o700, dir_fd=parent, follow_symlinks=False)
            root = os.open(stage, _CHECKOUT_DIR_FLAGS, dir_fd=parent)
        except BaseException:
            os.rmdir(stage, dir_fd=parent)
            raise
        published = False
        directories = [()]
        try:
            _materialize(lease, root, entries, meta, aliases, policy, directories)
            _require_absent(parent, destination.name)
            os.rename(stage, destination.name, src_dir_fd=parent, dst_dir_fd=parent)
            published = True
            os.fsync(parent)
        finally:
            try:
                if not published:
                    _discard_stage(parent, stage, root, directories)
            finally:
                os.close(root)
