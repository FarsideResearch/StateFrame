"""Host-owned workspace observations and interprocess writer fences."""
from __future__ import annotations

import errno
import fcntl
import io
import os
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, BinaryIO

from kernel_request_auth_identity import _installation_home
from kernel_secret_scan import SecretScanner
from kernel_workspace_index import (
    EntryFilter,
    StatRow,
    WorkspaceIndex,
    WorkspaceWork,
    _footprint,
    write_stat_index,
)
from kernel_workspace_objects import (
    MAX_OBJECT_BYTES,
    MAX_STREAM_BYTES,
    Digest,
    ObjectIntegrityError,
    ObjectLease,
    ObjectRef,
)
from kernel_workspace_tree import Directory, File, Symlink, lookup

if TYPE_CHECKING:
    from kernel_workspace_feed import WorkspaceFeed
    from kernel_workspace_refs import Ref, Rel


_EPOCH_BYTES = 16
_INLINE_SCAN_BYTES = 64 << 10


class WorkspaceUnavailable(OSError):
    """The host could not establish a current workspace observation."""


@dataclass(frozen=True, slots=True)
class WorkspaceFence:
    """A host-private lock shared by every reader and writer of one workcopy.

    The host must confine or reap other processes before relying on the fence.
    Writers reserve it before launch and hold it until all writes have stopped.
    """

    root: Path
    _lock_path: Path
    _lock_identity: tuple[int, int]

    @classmethod
    def create(cls, root: Path, private_directory: Path) -> WorkspaceFence:
        """Create a new fence in an exclusive, host-only directory."""
        root = Path(os.path.abspath(root))
        private_directory = private_directory.parent.resolve() / private_directory.name
        if private_directory.is_relative_to(root.resolve()):
            raise ValueError("workspace fence must be outside the workcopy")
        private_directory.mkdir(mode=0o700)
        path = private_directory / "fence"
        with _create_lock(path) as fd:
            _advance_epoch(fd)
            info = os.fstat(fd)
        return cls(root, path, (info.st_dev, info.st_ino))

    @contextmanager
    def _lock(self, mode: int) -> Iterator[tuple[int, bytes]]:
        with ExitStack() as stack:
            try:
                fd = os.open(self._lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
                stack.callback(os.close, fd)
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or info.st_uid != os.geteuid() or info.st_mode & 0o077
                        or (info.st_dev, info.st_ino) != self._lock_identity
                        or info.st_size != _EPOCH_BYTES):
                    raise OSError(errno.ESTALE, "invalid workspace fence")
                epoch = os.pread(fd, _EPOCH_BYTES, 0)
                if len(epoch) != _EPOCH_BYTES:
                    raise OSError(errno.EIO, "incomplete workspace epoch")
            except OSError as error:
                raise WorkspaceUnavailable("workspace fence unavailable") from error
            yield fd, epoch

    @contextmanager
    def writer(self) -> Iterator[tuple[bytes, bytes]]:
        """Reserve the workspace and revoke the preceding epoch before any write."""
        with self._lock(fcntl.LOCK_EX) as (fd, previous):
            try:
                epoch = _advance_epoch(fd)
            except OSError as error:
                raise WorkspaceUnavailable("workspace write epoch unavailable") from error
            yield previous, epoch


@contextmanager
def _create_lock(path: Path) -> Iterator[int]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        yield fd
    finally:
        os.close(fd)


def _advance_epoch(fd: int) -> bytes:
    epoch = os.urandom(_EPOCH_BYTES)
    if os.pwrite(fd, epoch, 0) != len(epoch):
        raise OSError(errno.EIO, "could not write workspace epoch")
    return epoch


class WorkspaceSnapshot:
    """An immutable tree and a read lease valid inside the pin context."""

    __slots__ = ("_active", "_boundary", "_epoch", "_pid", "_revocation", "_state", "_tree")

    def __init__(self, state: WorkspaceState, tree: Directory, epoch: bytes, boundary: str):
        self._state = state
        self._revocation = state._revocation
        self._tree = tree
        self._epoch = epoch
        self._boundary = boundary
        self._pid = os.getpid()
        self._active = True

    @property
    def tree(self) -> Directory:
        return self._tree

    @property
    def epoch(self) -> bytes:
        return self._epoch

    def validate_for_use(self, boundary: str) -> bool:
        if (not self._active or self._pid != os.getpid() or boundary != self._boundary
                or self._revocation != self._state._revocation):
            return False
        try:
            self._state._index.validate_root()
        except OSError:
            return False
        return True

    def persist(self, lease: ObjectLease) -> Digest:
        """Publish the admitted payloads and tree while the writer fence is held."""
        if not self.validate_for_use(self._boundary):
            raise WorkspaceUnavailable("workspace snapshot is no longer pinned")
        return self._state._persist(self._tree, lease)

    def persist_with_metadata(self, lease: ObjectLease) -> tuple[Digest, Ref]:
        """Publish the semantic tree and its separate restore metadata."""
        if not self.validate_for_use(self._boundary):
            raise WorkspaceUnavailable("workspace snapshot is no longer pinned")
        return self._state._persist_with_metadata(self._tree, lease)

    def check_retention(self) -> None:
        """Refuse an inadmissible projection before the caller writes any objects."""
        if not self.validate_for_use(self._boundary):
            raise WorkspaceUnavailable("workspace snapshot is no longer pinned")
        self._state._check_retention()

    def check_quiescent(self) -> None:
        """Reject feed-observed changes since the pinned observation."""
        self.check_retention()
        state = self._state
        if state._feed is not None and state._feed_current:
            update = state._feed.poll(writer_fenced=True)
            if update.requires_sweep or update.paths:
                state.invalidate()
                raise WorkspaceUnavailable("workspace changed during checkpoint capture")


_Paths = frozenset[bytes] | None
Coverage = Callable[[], tuple[object, EntryFilter]]
_Rows = dict[tuple[bytes, ...], StatRow | None]


class _SpoolSource(io.RawIOBase):
    def __init__(self, spool: BinaryIO | None, offset: int, size: int):
        super().__init__()
        self.spool = spool
        self.offset = offset
        self.remaining = size

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        if self.spool is None or not self.remaining:
            return 0
        data = os.pread(self.spool.fileno(), min(len(buffer), self.remaining), self.offset)
        buffer[:len(data)] = data
        self.remaining -= len(data)
        self.offset += len(data)
        return len(data)


def _spool_source(spool: BinaryIO | None, offset: int, size: int) -> BinaryIO:
    return io.BufferedReader(_SpoolSource(spool, offset, size))


class _PayloadSpool:
    """Keep each newly verified digest once in an anonymous observation spool."""

    def __init__(self, directory: Path, secrets: tuple[bytes, ...]):
        self.directory = directory
        self.secrets = secrets
        self.data: BinaryIO | None = None
        self.offset = 0
        self.start = 0
        self.positions: dict[bytes, tuple[int, int]] = {}
        self.rejected: set[bytes] = set()
        self.scanner: SecretScanner | None = None
        self._pending: bytes | None = None
        self._streaming = False
        self._resources = ExitStack()

    def append(self, block: bytes) -> None:
        # Small files fit in one read. The index supplies the digest of those
        # exact bytes at EOF, allowing repeats under this policy to skip scanning.
        if not self._streaming:
            if self._pending is None and len(block) <= _INLINE_SCAN_BYTES:
                self._pending = block
                return
            if self._pending is not None:
                self._append_scanned(self._pending)
                self._pending = None
            self._streaming = True
        self._append_scanned(block)

    def _append_scanned(self, block: bytes) -> None:
        if self.secrets:
            if self.scanner is None:
                self.scanner = SecretScanner(list(self.secrets))
            if self.scanner.feed(block):
                return
        if self.data is None:
            # The anonymous file spans pins until its tree is durable.
            self.data = self._resources.enter_context(tempfile.TemporaryFile(dir=self.directory))  # noqa: SIM115
        if self.data.write(block) != len(block):
            raise OSError("short workspace staging write")
        self.offset += len(block)

    def finish(self, digest: bytes, size: int) -> None:
        if self._pending is not None and digest not in self.positions and digest not in self.rejected:
            self._append_scanned(self._pending)
        self._pending = None
        self._streaming = False
        if self.scanner is not None and self.scanner.finish():
            self.rejected.add(digest)
        self.scanner = None
        if digest in self.rejected or digest in self.positions:
            if self.data is not None:
                self.data.seek(self.start)
                self.data.truncate()
            self.offset = self.start
        else:
            self.positions[digest] = self.start, size
            self.start = self.offset

    def publish(self, lease: ObjectLease, digests: set[bytes]) -> None:
        if self.data is not None:
            self.data.flush()
        small = []
        for digest in sorted(digests):
            try:
                offset, size = self.positions[digest]
            except KeyError as error:
                raise WorkspaceUnavailable("workspace payload observation is incomplete") from error
            if size <= MAX_OBJECT_BYTES:
                source = partial(_spool_source, self.data, offset, size)
                small.append((ObjectRef(digest.hex(), size), source))
            else:
                with _spool_source(self.data, offset, size) as source:
                    lease._publish_stream(source, size=size, max_bytes=MAX_STREAM_BYTES,
                                          expected=digest.hex())
        if small:
            lease.publish_many(small)

    def close(self) -> None:
        self._resources.close()


def _secret_policy() -> tuple[bytes, ...]:
    values: dict[bytes, None] = {}
    for name in ("kernel_checkpoint", "engine.kernel_checkpoint", "stateframe.engine.kernel_checkpoint"):
        checkpoint = sys.modules.get(name)
        values.update(dict.fromkeys(getattr(checkpoint, "_SECRETS", ())))
    return tuple(values)


class WorkspaceState:
    """Keep one reconciled index per host reader under a shared fence.

    Only the owner that completed a write can reuse its exact footprint.
    Another process observes the new epoch and reconciles conservatively.
    """

    def __init__(self, fence: WorkspaceFence, *, trust_stat: bool = False,
                 feed: WorkspaceFeed | None = None):
        self.fence = fence
        self._pid = os.getpid()
        self._mutex = RLock()
        self._index = WorkspaceIndex(fence.root, trust_stat=trust_stat)
        self._index.track_rows = True
        self._trust_stat = trust_stat
        self._feed = feed
        self._feed_current = False
        self._select: EntryFilter | None = None
        self._home_setting: Path | None = None
        self._home: Path | None = None
        self._secrets: tuple[bytes, ...] = ()
        self._spool: _PayloadSpool | None = None
        self._blocked: set[Rel] = set()
        self._leaders: dict[Rel, Rel] = {}
        self._external_links = False
        self._db_rows: _Rows = {}
        self._db_full = True
        self._db_stale = False
        self._meta_rows: _Rows = {}
        self._meta_full = True
        self._published: tuple[tuple[int, int], Directory] | None = None
        self._metadata: Ref | None = None
        self._coverage_token = os.urandom(16)
        self._work_id = sha256(os.fsencode(fence._lock_path)).hexdigest()
        self._epoch: bytes | None = None
        self._coverage: object = None
        self._pending: tuple[bytes, _Paths] | None = None
        self._invalidated = False
        self._revocation = 0

    @property
    def work(self) -> WorkspaceWork:
        return self._index.work

    def _check_process(self) -> None:
        if self._pid != os.getpid():
            raise WorkspaceUnavailable("workspace observation belongs to another process")

    def invalidate(self) -> None:
        """Discard local freshness after a reported mutation without a footprint receipt."""
        self._check_process()
        with self._mutex:
            self._invalidated = True
            self._revocation += 1

    @contextmanager
    def begin_write(self, paths: Iterable[str | bytes] | None = None) -> Iterator[None]:
        """Declare exact paths, or None for a writer with unbounded effects.

        Include renamed or replaced directories, both sides of moves, and all
        affected hardlink aliases. An exception discards the footprint.
        """
        self._check_process()
        if isinstance(paths, (str, bytes)):
            paths = (paths,)
        changed = (frozenset(b"/".join(parts) for parts in _footprint(paths))
                   if paths is not None else None)
        with self.fence.writer() as (previous, epoch), self._mutex:
            earlier = self._changes_since_pin(previous)
            self._pending = (epoch, None)
            yield
            if earlier is not None and changed is not None:
                self._pending = (epoch, earlier | changed)

    def _changes_since_pin(self, epoch: bytes) -> _Paths:
        if self._invalidated:
            return None
        if self._epoch == epoch:
            return frozenset()
        if self._pending is not None and self._pending[0] == epoch:
            return self._pending[1]
        return None

    def _reset_spool(self) -> None:
        if self._spool is not None:
            self._spool.close()
        self._spool = None

    def _observe(self, paths: _Paths, select: EntryFilter | None, *, opaque: bool) -> Directory:
        if opaque:
            self._reset_spool()
        self._spool = self._spool or _PayloadSpool(self.fence._lock_path.parent, self._secrets)
        self._index.stage_block = self._spool.append
        self._index.stage_file = self._spool.finish
        try:
            tree = self._index.refresh(paths, select=select, opaque_write=opaque)
        finally:
            self._index.stage_block = self._index.stage_file = None
        if self._index.full:
            self._db_rows.clear()
            self._meta_rows.clear()
            self._blocked.clear()
            self._db_full = self._meta_full = True
            self._record_aliases()
        self._db_rows.update(self._index.changes)
        self._meta_rows.update(self._index.changes)
        self._check_paths(self._index.changes)
        return tree

    def _record_aliases(self) -> None:
        aliases: dict[tuple[int, int], list[Rel]] = {}
        self._leaders.clear()
        self._external_links = False
        for path, row in self._index.rows.items():
            if isinstance(row.entry, File) and row.entry.hardlinked:
                aliases.setdefault((row.info.st_dev, row.info.st_ino), []).append(path)
        for group in aliases.values():
            leader = min(group)
            if len(group) != self._index.rows[leader].info.st_nlink:
                self._external_links = True
            self._leaders.update(dict.fromkeys(group, leader))

    def _check_paths(self, rows: _Rows) -> None:
        if not rows:
            return
        home = self._home
        for path, row in rows.items():
            self._blocked.discard(path)
            if row is None:
                continue
            target = row.entry.target if isinstance(row.entry, Symlink) else None
            if target is not None and home is not None:
                absolute = self.fence.root.joinpath(*map(os.fsdecode, path))
                if Path(os.path.realpath(absolute)).is_relative_to(home):
                    self._blocked.add(path)
            if self._secrets:
                for value in (b"/".join(path), target or b""):
                    scanner = SecretScanner(list(self._secrets))
                    if scanner.feed(value) or scanner.finish():
                        self._blocked.add(path)
            if (isinstance(row.entry, File) and self._spool is not None
                    and row.entry.sha256 in self._spool.rejected):
                self._blocked.add(path)

    def _protect_home(self, select: EntryFilter | None) -> EntryFilter | None:
        if self._home is None:
            return select
        root = self.fence.root
        if root == self._home or root.is_relative_to(self._home / "workspace-state"):
            raise WorkspaceUnavailable("workspace root overlaps private storage")
        if not self._home.is_relative_to(root):
            # An owned run may itself be under the installation's runs directory.
            return select
        prefix = tuple(map(os.fsencode, self._home.relative_to(root).parts))

        def admitted(path: Rel, directory: bool) -> bool:
            return path[:len(prefix)] != prefix and (select is None or select(path, directory))

        return admitted

    def _check_retention(self) -> None:
        if (self._blocked or self._external_links or _secret_policy() != self._secrets
                or _installation_home() != self._home_setting):
            raise WorkspaceUnavailable("workspace snapshot violates retention policy")

    def _persist(self, tree: Directory, lease: ObjectLease, *, verify_previous: bool = False) -> Digest:
        from kernel_workspace_tree_io import write_tree

        lease._check(write=True)
        self._check_retention()
        namespace = lease._store._objects_identity
        previous = self._published[1] if self._published is not None and self._published[0] == namespace else None
        if previous is not None:
            try:
                prior = lease.resolve(previous.digest.hex(), max_bytes=MAX_OBJECT_BYTES)
                if verify_previous and previous.digest == tree.digest:
                    lease.read(prior)
            except FileNotFoundError:
                previous = None
        if previous is None and self._published is not None:
            # A different namespace or an unretained prior root requires an
            # independent observation before a new self-contained publication.
            current = self._observe(None, self._select, opaque=True)
            if current.digest != tree.digest or self._blocked or self._external_links:
                self.invalidate()
                raise WorkspaceUnavailable("workspace changed during snapshot publication")
            self._metadata = None
        rows = self._index.rows if self._db_full or previous is None else self._db_rows
        if rows:
            digests = set()
            for path, row in rows.items():
                if row is None or not isinstance(row.entry, File):
                    continue
                old = lookup(previous, path) if previous is not None else None
                if not isinstance(old, File) or old.sha256 != row.entry.sha256:
                    digests.add(row.entry.sha256)
            if digests and self._spool is not None:
                self._spool.publish(lease, digests)
        oid = tree.digest.hex()
        if previous is None or previous.digest != tree.digest:
            parent_oid = previous.digest.hex() if previous is not None else None
            written = write_tree(lease, tree, previous=parent_oid)
            if written != oid:
                raise ObjectIntegrityError("workspace tree writer returned an inconsistent root")
        if self._db_rows or self._db_full:
            self._commit_index(lease, oid)
        self._published = namespace, tree
        self._reset_spool()
        return oid

    def _commit_index(self, lease: ObjectLease, oid: Digest) -> None:
        full = self._db_full or self._db_stale
        options = {"root": oid, "coverage": self._coverage_token, "trust_stat": self._trust_stat}
        try:
            if not full:
                write_stat_index(lease, self._work_id, self._db_rows, full=False, **options)
        except sqlite3.Error:
            full = True
        if full:
            try:
                write_stat_index(lease, self._work_id, self._index.rows, full=True, **options)
            except sqlite3.Error:
                # A damaged or busy disposable cache never certifies an object.
                self._db_stale = True
            else:
                self._db_stale = False
        self._db_rows.clear()
        self._db_full = False

    def _persist_with_metadata(self, tree: Directory, lease: ObjectLease) -> tuple[Digest, Ref]:
        from kernel_workspace_tree_io import write_restore_meta

        oid = self._persist(tree, lease, verify_previous=True)
        previous = self._metadata
        if previous is not None:
            try:
                lease.read(lease.resolve(previous.oid, max_bytes=MAX_OBJECT_BYTES))
            except FileNotFoundError:
                previous = None
        full = previous is None or self._meta_full
        rows = self._index.rows if full else self._meta_rows
        if rows or previous is None:
            self._metadata = write_restore_meta(
                lease, ((path, row.info.st_mtime_ns, self._leaders.get(path))
                        for path, row in rows.items() if row is not None),
                previous=None if full else previous,
                removed=(path for path, row in rows.items() if row is None),
            )
            self._meta_rows.clear()
            self._meta_full = False
        assert self._metadata is not None
        return oid, self._metadata

    @contextmanager
    def pin(self, boundary: str, *, coverage: Coverage | None = None,
            recheck: Iterable[bytes] = ()) -> Iterator[WorkspaceSnapshot]:
        """Observe at the boundary and hold the writer fence through its use."""
        self._check_process()
        with self.fence._lock(fcntl.LOCK_SH) as (_, epoch), self._mutex:
            try:
                self._index.validate_root(unchanged=False)
                key, select = coverage() if coverage is not None else (None, None)
                secrets = _secret_policy()
                setting = _installation_home()
                home_changed = setting != self._home_setting
                if home_changed:
                    self._home = Path(os.path.realpath(setting))
                    self._home_setting = setting
                paths = self._changes_since_pin(epoch)
                opaque = self._invalidated or (self._epoch is not None and paths is None)
                if key != self._coverage or secrets != self._secrets or home_changed:
                    # A new filter can admit previously unseen paths.
                    opaque = opaque or self._epoch is not None
                    paths = None
                    self._coverage_token = os.urandom(16)
                self._secrets = secrets
                select = self._protect_home(select)
                self._select = select
                feed = self._feed
                self._feed_current = False
                if feed is not None and paths is not None:
                    update = feed.poll(writer_fenced=True)
                    if update.requires_sweep:
                        paths, opaque = None, True
                    else:
                        paths = paths.union(update.paths)
                        self._feed_current = True
                if paths is not None:
                    paths = paths.union(recheck)
                watching = feed is not None and paths is None and feed.rebuild(writer_fenced=True)
                try:
                    tree = self._observe(paths, select, opaque=opaque)
                except OSError:
                    if paths is None:
                        raise
                    # A structural edit can invalidate a sparse footprint.
                    self._feed_current = False
                    watching = feed is not None and feed.rebuild(writer_fenced=True)
                    tree = self._observe(None, select, opaque=True)
                if watching and feed is not None:
                    update = feed.poll(writer_fenced=True, after_full_sweep=True)
                    if update.requires_sweep or update.paths:
                        raise WorkspaceUnavailable("workspace changed during fenced baseline")
                    self._feed_current = True
            except (OSError, ValueError, RuntimeError) as error:
                if self._feed is not None:
                    self._feed.close()
                self._reset_spool()
                self.invalidate()
                raise WorkspaceUnavailable("workspace observation unavailable") from error
            self._epoch = epoch
            self._coverage = key
            self._invalidated = False
            self._pending = None
            snapshot = WorkspaceSnapshot(self, tree, epoch, boundary)
            try:
                yield snapshot
            finally:
                snapshot._active = False
