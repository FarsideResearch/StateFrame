"""Immutable, canonical identities for one observed POSIX workspace tree.

The encoding hashes a versioned domain followed by length-prefixed fields.
Directory membership uses raw-byte radix pages: up to 16 names form a sorted
bucket, and larger pages split by the next byte with an optional terminal.
The shape depends only on membership. Update and diff can skip sibling pages.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import groupby
from operator import itemgetter
from typing import TypeVar

_DOMAIN = b"stateframe.workspace-tree/1\0"
_BUCKET_SIZE = 16
_KEY = itemgetter(0)
_Key = TypeVar("_Key", bytes, int)
_Value = TypeVar("_Value")


def _field(value: bytes) -> bytes:
    return len(value).to_bytes(4, "big") + value


def _preimage(tag: bytes, *fields: bytes) -> bytes:
    return b"".join((_DOMAIN, _field(tag), *map(_field, fields)))


def _uint(value: int) -> bytes:
    return value.to_bytes(8, "big")


def validate_name(name: bytes) -> None:
    if (not isinstance(name, bytes) or not name or len(name) > 255
            or name in (b".", b"..") or b"/" in name or b"\0" in name):
        raise ValueError("invalid workspace name")


@dataclass(frozen=True, slots=True)
class File:
    mode: int
    size: int
    sha256: bytes
    hardlinked: bool = False
    digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, bytes):
            raise TypeError("file SHA-256 must be bytes")
        if len(self.sha256) != 32:
            raise ValueError("invalid file SHA-256")
        object.__setattr__(self, "digest", sha256(self._preimage()).digest())

    def _preimage(self) -> bytes:
        return _preimage(b"file", _uint(self.mode), _uint(self.size), self.sha256,
                         bytes((self.hardlinked,)))


@dataclass(frozen=True, slots=True)
class Symlink:
    mode: int
    target: bytes
    digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.target, bytes):
            raise TypeError("symlink target must be bytes")
        object.__setattr__(self, "digest", sha256(self._preimage()).digest())

    def _preimage(self) -> bytes:
        return _preimage(b"symlink", _uint(self.mode), self.target)


@dataclass(frozen=True, slots=True)
class Special:
    mode: int
    kind: int
    device: int
    size: int
    digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", sha256(self._preimage()).digest())

    def _preimage(self) -> bytes:
        return _preimage(b"special", _uint(self.mode), _uint(self.kind), _uint(self.device),
                         _uint(self.size))


@dataclass(frozen=True, slots=True)
class _Page:
    digest: bytes
    count: int
    rows: tuple[tuple[bytes, Entry], ...] = ()
    terminal: tuple[bytes, Entry] | None = None
    edges: tuple[tuple[int, _Page], ...] = ()

    def _preimage(self) -> bytes:
        return (_bucket_preimage(self.rows) if self.rows
                else _branch_preimage(self.terminal, self.edges))


@dataclass(frozen=True, slots=True)
class Directory:
    mode: int
    _page: _Page | None = field(repr=False)
    digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", sha256(self._preimage()).digest())

    def _preimage(self) -> bytes:
        return _preimage(b"dir", _uint(self.mode), self._page.digest if self._page else b"")


Entry = File | Symlink | Special | Directory
_Row = tuple[bytes, Entry]


@dataclass(slots=True)
class TreeWork:
    pages_visited: int = 0
    pages_hashed: int = 0


def _encode_row(row: _Row) -> bytes:
    name, entry = row
    return _field(name) + entry.digest


def _bucket_preimage(rows: tuple[_Row, ...]) -> bytes:
    return _preimage(b"bucket", *map(_encode_row, rows))


def _branch_preimage(terminal: _Row | None, edges: tuple[tuple[int, _Page], ...]) -> bytes:
    return _preimage(b"branch", _encode_row(terminal) if terminal else b"",
                     *(bytes((slot,)) + page.digest for slot, page in edges))


def _bucket(rows: tuple[_Row, ...], work: TreeWork) -> _Page | None:
    if not rows:
        return None
    work.pages_hashed += 1
    return _Page(sha256(_bucket_preimage(rows)).digest(), len(rows), rows)


def _branch(terminal: _Row | None, edges: tuple[tuple[int, _Page], ...], work: TreeWork) -> _Page:
    work.pages_hashed += 1
    count = int(terminal is not None) + sum(page.count for _, page in edges)
    return _Page(
        sha256(_branch_preimage(terminal, edges)).digest(),
        count, terminal=terminal, edges=edges,
    )


def _build(rows: tuple[_Row, ...], depth: int, work: TreeWork) -> _Page | None:
    if len(rows) <= _BUCKET_SIZE:
        return _bucket(rows, work)
    terminal = rows[0] if len(rows[0][0]) == depth else None
    groups = groupby(rows[int(terminal is not None):], key=lambda row: row[0][depth])
    edges = []
    for slot, group in groups:
        page = _build(tuple(group), depth + 1, work)
        assert page is not None  # groupby produces nonempty groups
        edges.append((slot, page))
    return _branch(terminal, tuple(edges), work)


def directory(mode: int, entries: Iterable[_Row], work: TreeWork | None = None) -> Directory:
    rows = sorted(entries, key=_KEY)
    previous = None
    for name, _ in rows:
        validate_name(name)
        if name == previous:
            raise ValueError("duplicate workspace name")
        previous = name
    return Directory(mode, _build(tuple(rows), 0, work if work is not None else TreeWork()))


def _iterate(page: _Page | None, work: TreeWork) -> Iterator[_Row]:
    if page is None:
        return
    work.pages_visited += 1
    if page.rows:
        yield from page.rows
    else:
        if page.terminal is not None:
            yield page.terminal
        for _, child in page.edges:
            yield from _iterate(child, work)


def children(node: Directory) -> Iterator[_Row]:
    """List immediate children in unsigned raw filename order."""
    return _iterate(node._page, TreeWork())


def _get(node: Directory, name: bytes, work: TreeWork) -> Entry | None:
    page = node._page
    depth = 0
    while page is not None:
        work.pages_visited += 1
        if page.rows:
            index = bisect_left(page.rows, name, key=_KEY)
            return page.rows[index][1] if index < len(page.rows) and page.rows[index][0] == name else None
        if len(name) == depth:
            return page.terminal[1] if page.terminal is not None else None
        slot = name[depth]
        index = bisect_left(page.edges, slot, key=_KEY)
        if index == len(page.edges) or page.edges[index][0] != slot:
            return None
        page = page.edges[index][1]
        depth += 1
    return None


def lookup(node: Directory, path: tuple[bytes, ...], work: TreeWork | None = None) -> Entry | None:
    """Look up a validated relative path without following links."""
    work = work if work is not None else TreeWork()
    for name in path:
        validate_name(name)
    entry: Entry | None = node
    for name in path:
        if not isinstance(entry, Directory):
            return None
        entry = _get(entry, name, work)
    return entry


def _same(left: Entry | _Page | None, right: Entry | _Page | None) -> bool:
    return left is right or (left is not None and right is not None and left.digest == right.digest)


def _set(
    page: _Page | None, name: bytes, entry: Entry | None, depth: int, work: TreeWork,
) -> _Page | None:
    if page is not None:
        work.pages_visited += 1
    if page is None or page.rows:
        rows = dict(page.rows) if page is not None else {}
        if _same(rows.get(name), entry):
            return page
        if entry is None:
            del rows[name]
        else:
            rows[name] = entry
        return _build(tuple(sorted(rows.items())), depth, work)

    terminal = page.terminal
    edges = page.edges
    if len(name) == depth:
        if _same(terminal[1] if terminal else None, entry):
            return page
        terminal = (name, entry) if entry is not None else None
    else:
        slot = name[depth]
        index = bisect_left(edges, slot, key=_KEY)
        found = index < len(edges) and edges[index][0] == slot
        previous = edges[index][1] if found else None
        child = _set(previous, name, entry, depth + 1, work)
        if child is previous:
            return page
        replacement = ((slot, child),) if child is not None else ()
        if found:
            edges = edges[:index] + replacement + edges[index + 1:]
        else:
            edges = edges[:index] + replacement + edges[index:]
    if int(terminal is not None) + sum(child.count for _, child in edges) <= _BUCKET_SIZE:
        rows = [terminal] if terminal else []
        for _, child in edges:
            rows.extend(_iterate(child, work))
        return _bucket(tuple(rows), work)
    return _branch(terminal, edges, work)


def replace(node: Directory, name: bytes, entry: Entry | None, work: TreeWork | None = None) -> Directory:
    """Return a directory with one child added, replaced or removed."""
    validate_name(name)
    page = _set(node._page, name, entry, 0, work if work is not None else TreeWork())
    return node if page is node._page else Directory(node.mode, page)


def with_mode(node: Directory, mode: int) -> Directory:
    return node if node.mode == mode else Directory(mode, node._page)


def _join(
    left: Iterable[tuple[_Key, _Value]], right: Iterable[tuple[_Key, _Value]],
) -> Iterator[tuple[_Key, _Value | None, _Value | None]]:
    left_items, right_items = iter(left), iter(right)
    old, new = next(left_items, None), next(right_items, None)
    while old is not None or new is not None:
        if old is not None and (new is None or old[0] < new[0]):
            yield old[0], old[1], None
            old = next(left_items, None)
        elif new is not None and (old is None or new[0] < old[0]):
            yield new[0], None, new[1]
            new = next(right_items, None)
        else:
            assert old is not None and new is not None
            yield old[0], old[1], new[1]
            old, new = next(left_items, None), next(right_items, None)


def _changed_pages(
    old: _Page | None, new: _Page | None, work: TreeWork,
) -> Iterator[tuple[bytes, Entry | None, Entry | None]]:
    if _same(old, new):
        return
    if old is not None and new is not None and old.edges and new.edges:
        work.pages_visited += 2
        yield from _join((old.terminal,) if old.terminal else (), (new.terminal,) if new.terminal else ())
        for _, before, after in _join(old.edges, new.edges):
            yield from _changed_pages(before, after, work)
    else:
        yield from _join(_iterate(old, work), _iterate(new, work))


@dataclass(frozen=True, slots=True)
class Change:
    path: tuple[bytes, ...]
    before: Entry | None
    after: Entry | None


def diff(old: Directory, new: Directory, work: TreeWork | None = None) -> Iterator[Change]:
    """Report changed entries in raw path order, including added/removed descendants."""
    work = work if work is not None else TreeWork()
    pending: list[tuple[tuple[bytes, ...], Entry | None, Entry | None]] = [((), old, new)]
    while pending:
        path, before, after = pending.pop()
        if _same(before, after):
            continue
        old_dir = before if isinstance(before, Directory) else None
        new_dir = after if isinstance(after, Directory) else None
        if old_dir is None or new_dir is None or old_dir.mode != new_dir.mode:
            yield Change(path, before, after)
        if old_dir is not None or new_dir is not None:
            changed = list(_changed_pages(
                old_dir._page if old_dir else None, new_dir._page if new_dir else None, work,
            ))
            pending.extend(((*path, name), a, b) for name, a, b in reversed(changed))
