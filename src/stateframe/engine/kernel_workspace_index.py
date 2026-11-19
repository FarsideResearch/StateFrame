"""Host-side POSIX stat index for a quiescent workspace.

The caller owns the writer fence. A sparse refresh requires a complete footprint,
including structurally changed directories, both sides of moves, and hardlink
aliases. Use a full sweep for an unknown footprint. The index retains identities
and digests in memory. Published trees confer neither a lease nor a byte seal.
"""
from __future__ import annotations

import errno
import fcntl
import os
import secrets
import sqlite3
import stat
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass
from hashlib import sha256
from time import time_ns
from typing import TYPE_CHECKING
from weakref import finalize

from kernel_workspace_tree import (
    Directory,
    Entry,
    File,
    Special,
    Symlink,
    TreeWork,
    children,
    directory,
    lookup,
    replace,
    validate_name,
    with_mode,
)

if TYPE_CHECKING:
    from kernel_workspace_objects import ObjectLease


# Mirror the live projection's racy timestamp guard. Measure from the start of
# observation so a slow read cannot age a same-tick identity into a hit.
_RACY_NS = 2_000_000_000
_MEMO_LIMIT = 200_000
_MAX_DEPTH = 256
_CHUNK_BYTES = 256 * 1024
_StatKey = tuple[int, int, int, int, int, int, int]
_RelativePath = tuple[bytes, ...]
_Path = str | bytes | os.PathLike[str] | os.PathLike[bytes]
EntryFilter = Callable[[_RelativePath, bool], bool]


@dataclass(frozen=True, slots=True)
class StatRow:
    info: os.stat_result
    entry: Entry
    verified_at: int
    racy: bool


@dataclass(slots=True)
class WorkspaceWork(TreeWork):
    content_reads: int = 0
    rehashed_bytes: int = 0
    stat_hits: int = 0
    directories_scanned: int = 0


def _signature(info: os.stat_result) -> _StatKey:
    return (
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
        info.st_ctime_ns, info.st_mode, info.st_nlink,
    )


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _check_stat(before: os.stat_result, after: os.stat_result) -> None:
    if _signature(before) != _signature(after):
        raise OSError(errno.ESTALE, "workspace entry changed during observation")


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@contextmanager
def _open_directory(parent: int, name: bytes) -> Iterator[int]:
    descriptor = os.open(name, _directory_flags(), dir_fd=parent)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _open_root(components: _RelativePath) -> Iterator[int]:
    descriptor = os.open(b"/", _directory_flags())
    try:
        for name in components:
            child = os.open(name, _directory_flags(), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _footprint(paths: Iterable[_Path]) -> list[_RelativePath]:
    unique = set()
    for path in paths:
        parts = tuple(os.fsencode(path).split(b"/"))
        if len(parts) > _MAX_DEPTH:
            raise OSError(errno.ENAMETOOLONG, "workspace depth limit exceeded")
        for name in parts:
            validate_name(name)
        unique.add(parts)
    result = []
    for parts in sorted(unique):
        # A recursively refreshed ancestor covers any subsequent descendants.
        if not result or parts[:len(result[-1])] != result[-1]:
            result.append(parts)
    return result


class WorkspaceIndex:
    """Incrementally refresh a complete tree under the caller's fence.

    Stat reuse is opt-in for trusted, locally coherent timestamps. Initialize
    with a full refresh. Each successful refresh publishes one immutable tree.
    Use a new index when rebinding to a different physical root.
    """

    def __init__(self, root: _Path, *, trust_stat: bool = False):
        absolute = os.path.abspath(os.fsencode(root))
        self._components = tuple(name for name in absolute.split(b"/") if name)
        self._trust_stat = trust_stat
        self._memo: OrderedDict[_StatKey, tuple[bytes, int]] = OrderedDict()
        self._identity: tuple[int, int] | None = None
        self._root_pin: finalize | None = None
        self._root_info: os.stat_result | None = None
        self._last_observation_ns: int | None = None
        self._root: Directory | None = None
        self._select: EntryFilter | None = None
        self.track_rows = False
        self.stage_block: Callable[[bytes], None] | None = None
        self.stage_file: Callable[[bytes, int], None] | None = None
        self.rows: dict[_RelativePath, StatRow] = {}
        self.changes: dict[_RelativePath, StatRow | None] = {}
        self._racy: set[_RelativePath] = set()
        self._sparse = False
        self.full = False
        self.work = WorkspaceWork()

    @property
    def root(self) -> Directory:
        if self._root is None:
            raise RuntimeError("workspace index requires a full refresh")
        return self._root

    def validate_root(self, *, unchanged: bool = True) -> None:
        """Reopen the full ancestor chain and verify the physical binding."""
        with _open_root(self._components) as root_fd:
            current = os.fstat(root_fd)
            if self._identity is not None and _identity(current) != self._identity:
                raise OSError(errno.ESTALE, "workspace root was replaced")
            if unchanged:
                if self._root_info is None:
                    raise OSError(errno.ESTALE, "workspace root has no observation")
                _check_stat(self._root_info, current)

    def refresh(
        self, changed: Iterable[_Path] | None = None, *,
        select: EntryFilter | None = None, opaque_write: bool = False,
    ) -> Directory:
        """Sweep or apply an exact footprint under the caller's writer fence.

        A different selection requires a full sweep. An opaque write invalidates
        digest reuse even on a filesystem with trusted stat tuples.
        """
        self.work = WorkspaceWork()
        self.changes = {}
        self.full = False
        self._select = select
        if opaque_write:
            self._memo.clear()
        try:
            return self._refresh(changed, opaque_write=opaque_write)
        except BaseException:
            self._memo.clear()
            self.changes = {}
            raise
        finally:
            self._select = None
            self._sparse = False

    def _refresh(self, changed: Iterable[_Path] | None, *, opaque_write: bool) -> Directory:
        observed_at = time_ns()
        if self._last_observation_ns is not None and observed_at < self._last_observation_ns:
            self._memo.clear()
            opaque_write = True
        paths = None
        if changed is not None and not opaque_write:
            paths = _footprint((*changed, *(b"/".join(parts) for parts in self._racy)))
        if paths == []:
            self.validate_root()
            return self.root
        if paths is not None:
            candidate = self.root
            self._sparse = True

        self._last_observation_ns = observed_at

        with _open_root(self._components) as root_fd:
            info = os.fstat(root_fd)
            identity = _identity(info)
            if self._identity is not None and identity != self._identity:
                raise OSError(errno.ESTALE, "workspace root was replaced")
            if paths is None:
                candidate = self._scan(root_fd, info, (), observed_at, dirty=opaque_write)
            else:
                for parts in paths:
                    if self.track_rows:
                        self._remove_rows(parts, lookup(candidate, parts))
                    candidate = self._refresh_path(root_fd, candidate, parts, 0, observed_at)
                candidate = with_mode(candidate, stat.S_IMODE(info.st_mode))
                self._note((), info, candidate, observed_at)
            _check_stat(info, os.fstat(root_fd))
            with _open_root(self._components) as current_root:
                _check_stat(info, os.fstat(current_root))
            if self._root_pin is None:
                # Keep the original inode allocated for this index's lifetime,
                # so a deleted root cannot be mistaken for an inode reuse.
                self._root_pin = finalize(self, os.close, os.dup(root_fd))

        self._identity = identity
        self._root_info = info
        self._root = candidate
        if self.track_rows:
            self.full = paths is None
            if self.full:
                self.rows.clear()
                self._racy.clear()
            for parts, row in self.changes.items():
                self._racy.discard(parts)
                if row is None:
                    self.rows.pop(parts, None)
                else:
                    self.rows[parts] = row
                    if row.racy:
                        self._racy.add(parts)
        return candidate

    def _remove_rows(self, path: _RelativePath, entry: Entry | None) -> None:
        if entry is None:
            return
        self.changes[path] = None
        if isinstance(entry, File) and entry.hardlinked:
            raise OSError(errno.ESTALE, "workspace footprint may omit hardlink aliases")
        if isinstance(entry, Directory):
            for name, child in children(entry):
                self._remove_rows((*path, name), child)

    def _note(
        self, parts: _RelativePath, info: os.stat_result, entry: Entry, verified_at: int,
        *, racy: bool = False,
    ) -> Entry:
        if self.track_rows:
            self.changes[parts] = StatRow(info, entry, verified_at, racy)
        return entry

    def _scan(
        self, fd: int, before: os.stat_result, prefix: _RelativePath, observed_at: int, *, dirty: bool,
    ) -> Directory:
        self.work.directories_scanned += 1
        rows = []
        with os.scandir(fd) as entries:
            for item in entries:
                name = os.fsencode(item.name)
                validate_name(name)
                info = item.stat(follow_symlinks=False)
                parts = (*prefix, name)
                if not self._included(parts, info):
                    continue
                entry = self._observe(fd, parts, info, observed_at, dirty=dirty)
                rows.append((name, entry))
        _check_stat(before, os.fstat(fd))
        result = directory(stat.S_IMODE(before.st_mode), rows, self.work)
        self._note(prefix, before, result, observed_at)
        return result

    def _included(self, parts: _RelativePath, info: os.stat_result) -> bool:
        return self._select is None or self._select(parts, stat.S_ISDIR(info.st_mode))

    def _refresh_path(
        self, fd: int, node: Directory, parts: _RelativePath, depth: int, observed_at: int,
    ) -> Directory:
        name = parts[depth]
        if depth + 1 == len(parts):
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                entry = None
            else:
                entry = (self._observe(fd, parts, info, observed_at, dirty=True)
                         if self._included(parts, info) else None)
        else:
            prefix = parts[:depth + 1]
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if not self._included(prefix, before):
                raise OSError(errno.ESTALE, "workspace footprint omits a coverage change")
            previous = lookup(node, (name,), self.work)
            if not isinstance(previous, Directory):
                raise OSError(errno.ESTALE, "workspace footprint omits a changed ancestor")
            with _open_directory(fd, name) as child_fd:
                _check_stat(before, os.fstat(child_fd))
                entry = self._refresh_path(child_fd, previous, parts, depth + 1, observed_at)
                _check_stat(before, os.fstat(child_fd))
            _check_stat(before, os.stat(name, dir_fd=fd, follow_symlinks=False))
            entry = with_mode(entry, stat.S_IMODE(before.st_mode))
            self._note(prefix, before, entry, observed_at)
        return replace(node, name, entry, self.work)

    def _observe(
        self, fd: int, parts: _RelativePath, before: os.stat_result, observed_at: int, *, dirty: bool,
    ) -> Entry:
        if len(parts) > _MAX_DEPTH:
            raise OSError(errno.ENAMETOOLONG, "workspace depth limit exceeded")
        name = parts[-1]
        mode = stat.S_IMODE(before.st_mode)
        kind = stat.S_IFMT(before.st_mode)
        if kind == stat.S_IFREG:
            if self.track_rows and self._sparse and before.st_nlink > 1:
                raise OSError(errno.ESTALE, "workspace footprint may omit hardlink aliases")
            digest, verified_at = self._file_digest(fd, name, before, observed_at, dirty=dirty)
            entry = File(mode, before.st_size, digest, before.st_nlink > 1)
            racy = self._trust_stat and not self._stable(before, verified_at)
            return self._note(parts, before, entry, verified_at, racy=racy)
        if kind == stat.S_IFDIR:
            with _open_directory(fd, name) as child_fd:
                _check_stat(before, os.fstat(child_fd))
                entry = self._scan(child_fd, before, parts, observed_at, dirty=dirty)
        elif kind == stat.S_IFLNK:
            entry = Symlink(mode, os.readlink(name, dir_fd=fd))
        else:
            entry = Special(mode, kind, before.st_rdev, before.st_size)
        _check_stat(before, os.stat(name, dir_fd=fd, follow_symlinks=False))
        return self._note(parts, before, entry, observed_at)

    @staticmethod
    def _stable(info: os.stat_result, verified_at: int) -> bool:
        return max(info.st_mtime_ns, info.st_ctime_ns) < verified_at - _RACY_NS

    def _file_digest(
        self, parent: int, name: bytes, before: os.stat_result, observed_at: int, *, dirty: bool,
    ) -> tuple[bytes, int]:
        key = _signature(before)
        safe = self._trust_stat and self._stable(before, observed_at)
        if dirty or not safe:
            self._memo.pop(key, None)
        else:
            digest = self._memo.get(key)
            if digest is not None:
                _check_stat(before, os.stat(name, dir_fd=parent, follow_symlinks=False))
                self._memo.move_to_end(key)
                self.work.stat_hits += 1
                return digest

        # Verify the descriptor before any content read. NOFOLLOW refuses a
        # swapped link and NONBLOCK avoids hanging on a swapped FIFO.
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            _check_stat(before, os.fstat(fd))
            self.work.content_reads += 1
            digest_state = sha256()
            remaining = before.st_size
            while True:
                data = os.read(fd, min(_CHUNK_BYTES, remaining + 1))
                if not data:
                    break
                self.work.rehashed_bytes += len(data)
                remaining -= len(data)
                if remaining < 0:
                    raise OSError(errno.ESTALE, "workspace file grew during observation")
                digest_state.update(data)
                if self.stage_block is not None:
                    self.stage_block(data)
            if remaining:
                raise OSError(errno.ESTALE, "workspace file shrank during observation")
            _check_stat(before, os.fstat(fd))
        finally:
            os.close(fd)
        _check_stat(before, os.stat(name, dir_fd=parent, follow_symlinks=False))
        digest = digest_state.digest()
        if self.stage_file is not None:
            self.stage_file(digest, before.st_size)
        if safe:
            self._memo[key] = (digest, observed_at)
            if len(self._memo) > _MEMO_LIMIT:
                self._memo.popitem(last=False)
        return digest, observed_at


def open_workspace_index(
    root: _Path, *, enabled: bool = False, trust_stat: bool = False,
) -> WorkspaceIndex | None:
    """Construct an index only when its host explicitly opts in."""
    if enabled:
        return WorkspaceIndex(root, trust_stat=trust_stat)
    return None


def write_stat_index(
    lease: ObjectLease, work_id: str, rows: dict[_RelativePath, StatRow | None], *,
    full: bool, root: str, coverage: bytes, trust_stat: bool,
) -> None:
    """Commit disposable stat rows after the associated objects are durable."""
    from kernel_workspace_objects import _child_directory, _lock, _private_directory

    lease._check(write=True)
    with _child_directory(lease._namespace, "index", create=True) as directory:
        _private_directory(directory)
        with _lock(directory, fcntl.LOCK_EX):
            name = work_id + ".sqlite"
            target = ".index-" + secrets.token_hex(16) if full else name
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if full:
                flags |= os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(target, flags, 0o600, dir_fd=directory)
            except FileNotFoundError as error:
                raise sqlite3.DatabaseError("workspace stat database is missing") from error
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600):
                    raise sqlite3.DatabaseError("invalid workspace stat database")
                # SQLite needs a path for its rollback journal. The validated
                # directory is host-private and held through the transaction.
                path = lease._store._path / "index" / target
                with closing(sqlite3.connect(path)) as database, database:
                    database.execute("BEGIN IMMEDIATE")
                    if full:
                        _init_stat_database(database)
                    _update_stat_database(database, rows, root, coverage, trust_stat)
                if full:
                    # A replacement database must never replay its predecessor's
                    # abandoned rollback journal.
                    with suppress(FileNotFoundError):
                        os.unlink(name + "-journal", dir_fd=directory)
                    os.replace(target, name, src_dir_fd=directory, dst_dir_fd=directory)
                    os.fsync(directory)
            finally:
                os.close(fd)
                if full:
                    with suppress(FileNotFoundError):
                        os.unlink(target, dir_fd=directory)


def _init_stat_database(database: sqlite3.Connection) -> None:
    database.execute("""CREATE TABLE entries (
        path BLOB PRIMARY KEY, parent BLOB, dev INTEGER, ino INTEGER,
        size INTEGER, mtime INTEGER, ctime INTEGER, mode INTEGER, links INTEGER,
        verified INTEGER, racy INTEGER, entry BLOB, blob BLOB
    ) WITHOUT ROWID""")
    database.execute("CREATE INDEX parents ON entries (parent)")
    database.execute("CREATE INDEX aliases ON entries (dev, ino) WHERE links > 1")
    database.execute("CREATE INDEX racy_paths ON entries (racy) WHERE racy = 1")
    database.execute("""CREATE TABLE head (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        codec INTEGER, dev INTEGER, ino INTEGER, coverage BLOB, trusted INTEGER, tree TEXT
    )""")


def _update_stat_database(
    database: sqlite3.Connection, rows: dict[_RelativePath, StatRow | None],
    root: str, coverage: bytes, trust_stat: bool,
) -> None:
    database.executemany("DELETE FROM entries WHERE path = ?",
                         ((b"/".join(path),) for path, row in rows.items() if row is None))
    database.executemany("INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (_stat_values(path, row) for path, row in rows.items() if row is not None))
    identity = database.execute("SELECT dev, ino FROM entries WHERE path = x''").fetchone()
    if identity is None:
        raise sqlite3.DatabaseError("workspace stat database has no root row")
    database.execute("INSERT OR REPLACE INTO head VALUES (1,1,?,?,?,?,?)",
                     (*identity, coverage, trust_stat, root))


def _stat_values(path: _RelativePath, row: StatRow) -> tuple[object, ...]:
    info = row.info
    return (b"/".join(path), b"/".join(path[:-1]), info.st_dev, info.st_ino,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode, info.st_nlink,
            row.verified_at, row.racy, row.entry.digest,
            row.entry.sha256 if isinstance(row.entry, File) else None)
