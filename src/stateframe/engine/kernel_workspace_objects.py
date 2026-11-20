"""Host-private content objects for verified workspace snapshots."""

from __future__ import annotations

import fcntl
import hashlib
import io
import os
import re
import secrets
import stat
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, closing, contextmanager, suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import BinaryIO

MAX_OBJECT_BYTES = 16 << 20
MAX_STREAM_BYTES = (1 << 63) - 1
_CHUNK_BYTES = 1 << 20
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_NEW_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
_DIGEST = re.compile(r"[0-9a-f]{64}")
_NAMESPACE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}")
_STAGE = ".stage-"
_INDEX_STAGE = re.compile(r"\.index-[0-9a-f]{32}(?:-journal)?")


class ObjectIntegrityError(ValueError):
    """A reference, directory, or payload failed an integrity check."""


@dataclass(frozen=True, slots=True)
class ObjectRef:
    digest: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _DIGEST.fullmatch(self.digest):
            raise ObjectIntegrityError("invalid workspace object digest")
        if type(self.size) is not int or not 0 <= self.size <= MAX_OBJECT_BYTES:
            raise ObjectIntegrityError("invalid workspace object size")


@dataclass(frozen=True, slots=True)
class LargeObjectRef:
    digest: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.digest, str) or not _DIGEST.fullmatch(self.digest):
            raise ObjectIntegrityError("invalid workspace object digest")
        if type(self.size) is not int or not MAX_OBJECT_BYTES < self.size <= MAX_STREAM_BYTES:
            raise ObjectIntegrityError("invalid large workspace object size")


StoredRef = ObjectRef | LargeObjectRef
Digest = str
ObjectSource = Callable[[], BinaryIO]


def _ref(digest: Digest, size: int) -> StoredRef:
    return ObjectRef(digest, size) if size <= MAX_OBJECT_BYTES else LargeObjectRef(digest, size)


def _private_directory(fd: int) -> tuple[int, int]:
    info = os.fstat(fd)
    if info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ObjectIntegrityError("workspace object directory must be owner-private")
    return info.st_dev, info.st_ino


@contextmanager
def _directory(path: Path) -> Iterator[int]:
    """Open each component without following links, including ancestors."""
    path = path.absolute()
    if ".." in path.parts:
        raise ValueError("parent components are not allowed in object paths")
    fd = os.open(path.anchor, _DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _child_directory(parent: int, name: str, *, create: bool = False) -> Iterator[int]:
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
        except FileExistsError:
            pass
        else:
            # mkdir modes are filtered by umask; private store directories
            # need owner access to initialize their children and registries.
            os.chmod(name, 0o700, dir_fd=parent, follow_symlinks=False)
    fd = os.open(name, _DIR_FLAGS, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _lock(fd: int, operation: int) -> Iterator[None]:
    fcntl.flock(fd, operation)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _stream_digest(
    size: int,
    read: Callable[[int], bytes],
    output: BinaryIO | None = None,
) -> str:
    digest = hashlib.sha256()
    remaining = size
    while block := read(min(_CHUNK_BYTES, remaining + 1)):
        remaining -= len(block)
        if remaining < 0:
            raise ObjectIntegrityError("workspace object exceeds its declared size")
        digest.update(block)
        if output is not None and output.write(block) != len(block):
            raise OSError("short workspace object write")
    if remaining:
        raise ObjectIntegrityError("workspace object is shorter than its declared size")
    return digest.hexdigest()


def _copy_checked(
    ref: StoredRef, read: Callable[[int], bytes], output: BinaryIO | None = None,
) -> None:
    if _stream_digest(ref.size, read, output) != ref.digest:
        raise ObjectIntegrityError("workspace object content does not match its reference")


def _object_size(info: os.stat_result) -> int:
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o400):
        raise ObjectIntegrityError("invalid workspace object type, owner, or mode")
    return info.st_size


def _read_checked(directory: int, ref: StoredRef, output: BinaryIO | None = None) -> None:
    fd = os.open(ref.digest, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        before = os.fstat(fd)
        if _object_size(before) != ref.size:
            raise ObjectIntegrityError("invalid workspace object size")
        _copy_checked(ref, partial(os.read, fd), output)
        after = os.fstat(fd)
        # Link count can change as another publisher finishes staging recovery.
        if (_object_size(after), after.st_mtime_ns) != (ref.size, before.st_mtime_ns):
            raise ObjectIntegrityError("workspace object changed while being read")
    finally:
        os.close(fd)


def _stage_object(directory: int, ref: ObjectRef, source: ObjectSource) -> None:
    name = _STAGE + ref.digest
    # A previous publisher might have died before linking this staged file.
    with suppress(FileNotFoundError):
        os.unlink(name, dir_fd=directory)
    fd = os.open(name, _NEW_FILE_FLAGS, 0o600, dir_fd=directory)
    try:
        with os.fdopen(fd, "wb") as output, closing(source()) as incoming:
            _copy_checked(ref, incoming.read, output)
            output.flush()
            os.fchmod(fd, 0o400)
            os.fsync(fd)
    except BaseException:
        os.unlink(name, dir_fd=directory)
        raise


class ObjectLease:
    """A single-process, single-thread view held until the caller drops its pin."""

    def __init__(self, directory: int, *, writable: bool, namespace: int, store: WorkspaceObjects):
        self._directory = directory
        self._namespace = namespace
        self._store = store
        self._owner = os.getpid(), threading.get_ident()
        self._active = True
        self._writable = writable

    def _check(self, *, write: bool = False) -> int:
        if not self._active or self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError("workspace object lease is closed or belongs to another owner")
        if write and not self._writable:
            raise RuntimeError("workspace object collection lease is read-only")
        return self._directory

    def publish_many(self, entries: Iterable[tuple[ObjectRef, ObjectSource]]) -> None:
        """Make each digest durable before returning.

        Supply approved references from a verified snapshot and approved, bounded
        byte sources. A source can remain unopened when its object is present.
        Hold this lease through durable publication of the caller's root.
        """
        directory = self._check(write=True)
        seen: dict[str, int] = {}
        new: list[str] = []
        pending: list[str] = []
        with _lock(directory, fcntl.LOCK_EX):
            for ref, source in entries:
                if ref.size > MAX_OBJECT_BYTES:
                    raise ObjectIntegrityError("workspace object batch exceeds per-object limit")
                if ref.digest in seen:
                    if seen[ref.digest] != ref.size:
                        raise ObjectIntegrityError("conflicting workspace object sizes")
                    continue
                seen[ref.digest] = ref.size
                try:
                    _read_checked(directory, ref)
                except FileNotFoundError:
                    _stage_object(directory, ref, source)
                    new.append(ref.digest)
                else:
                    try:
                        os.stat(_STAGE + ref.digest, dir_fd=directory, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        pending.append(ref.digest)

            if new:
                # Persist the staging names before any canonical link. Their
                # presence records a directory sync still owed after interruption.
                os.fsync(directory)
                for digest in new:
                    os.link(_STAGE + digest, digest, src_dir_fd=directory,
                            dst_dir_fd=directory, follow_symlinks=False)
            if new or pending:
                os.fsync(directory)
                for digest in (*new, *pending):
                    os.unlink(_STAGE + digest, dir_fd=directory)

    def read(self, ref: ObjectRef) -> bytes:
        """Return bounded, digest-verified payload bytes while the lease is held."""
        if ref.size > MAX_OBJECT_BYTES:
            raise ObjectIntegrityError("workspace object exceeds bounded read limit")
        output = io.BytesIO()
        _read_checked(self._check(), ref, output)
        return output.getvalue()

    def resolve(self, oid: Digest, *, max_bytes: int) -> StoredRef:
        """Resolve a bounded name and length. Payload readers verify its digest."""
        directory = self._check()
        if not isinstance(oid, str) or not _DIGEST.fullmatch(oid):
            raise ObjectIntegrityError("invalid workspace object digest")
        size = _object_size(os.stat(oid, dir_fd=directory, follow_symlinks=False))
        if size > max_bytes:
            raise ObjectIntegrityError("workspace object exceeds its permitted size")
        return _ref(oid, size)

    def copy_verified(self, ref: StoredRef, output: BinaryIO, *, max_bytes: int) -> None:
        if ref.size > max_bytes:
            raise ObjectIntegrityError("workspace object exceeds its permitted size")
        _read_checked(self._check(), ref, output)

    def publish_stream(self, source: BinaryIO, *, size: int, max_bytes: int) -> StoredRef:
        """Stream already admitted bytes into a durable, deduplicated object."""
        return self._publish_stream(source, size=size, max_bytes=max_bytes)

    def _publish_stream(
        self, source: BinaryIO, *, size: int, max_bytes: int, expected: Digest | None = None,
    ) -> StoredRef:
        directory = self._check(write=True)
        if type(size) is not int or not 0 <= size <= min(max_bytes, MAX_STREAM_BYTES):
            raise ObjectIntegrityError("invalid workspace object stream size")
        temporary = ".incoming-" + secrets.token_hex(16)
        # The lease excludes GC while the stream is being ingested. Only final
        # link publication takes the batch lock, so long streams can overlap.
        fd = os.open(temporary, _NEW_FILE_FLAGS, 0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, "wb") as output:
                ref = _ref(_stream_digest(size, source.read, output), size)
                if expected is not None and ref.digest != expected:
                    raise ObjectIntegrityError("staged workspace payload changed")
                output.flush()
                with _lock(directory, fcntl.LOCK_EX):
                    marker = _STAGE + ref.digest
                    try:
                        _read_checked(directory, ref)
                    except FileNotFoundError:
                        os.fchmod(fd, 0o400)
                        os.fsync(fd)
                        os.replace(temporary, marker, src_dir_fd=directory, dst_dir_fd=directory)
                        os.fsync(directory)
                        os.link(marker, ref.digest, src_dir_fd=directory,
                                dst_dir_fd=directory, follow_symlinks=False)
                    try:
                        os.stat(marker, dir_fd=directory, follow_symlinks=False)
                    except FileNotFoundError:
                        pass
                    else:
                        os.fsync(directory)
                        os.unlink(marker, dir_fd=directory)
                return ref
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory)

    def restore(
        self,
        ref: ObjectRef,
        destination: Path,
        *,
        mode: int = 0o600,
        mtime_ns: int | None = None,
    ) -> None:
        """Copy to an absent entry in a caller-owned private staging directory."""
        store_directory = self._check()
        if ref.size > MAX_OBJECT_BYTES:
            raise ObjectIntegrityError("workspace object exceeds bounded restore limit")
        destination = Path(destination)
        if not destination.name or type(mode) is not int or not 0 <= mode <= 0o777:
            raise ValueError("invalid workspace restore destination or mode")
        with _directory(destination.parent) as directory:
            _private_directory(directory)
            temporary = ".restore-" + secrets.token_hex(16)
            fd = os.open(temporary, _NEW_FILE_FLAGS, 0o600, dir_fd=directory)
            try:
                with os.fdopen(fd, "wb") as output:
                    _read_checked(store_directory, ref, output)
                    output.flush()
                    os.fchmod(fd, mode)
                    if mtime_ns is not None:
                        os.utime(fd, ns=(mtime_ns, mtime_ns))
                    os.fsync(fd)
                # Publish without replacing another entry. The writable inode
                # belongs solely to this restore and has no link to the store.
                os.link(temporary, destination.name, src_dir_fd=directory,
                        dst_dir_fd=directory, follow_symlinks=False)
            finally:
                os.unlink(temporary, dir_fd=directory)
            os.fsync(directory)


@dataclass(frozen=True, slots=True)
class WorkspaceObjects:
    """One private retention and trust namespace on a local filesystem."""

    _path: Path
    _identity: tuple[int, int]
    _objects_identity: tuple[int, int]
    _refs_identity: tuple[int, int]

    @contextmanager
    def _lease(self, *, exclusive: bool) -> Iterator[ObjectLease]:
        with _directory(self._path) as root:
            if _private_directory(root) != self._identity:
                raise ObjectIntegrityError("workspace object namespace was replaced")
            operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            with _lock(root, operation), _child_directory(root, "objects") as directory:
                if _private_directory(directory) != self._objects_identity:
                    raise ObjectIntegrityError("workspace object directory was replaced")
                lease = ObjectLease(directory, writable=not exclusive, namespace=root, store=self)
                try:
                    yield lease
                finally:
                    lease._active = False

    def lease(self) -> AbstractContextManager[ObjectLease]:
        """Pin reads, publication and root commits against concurrent collection."""
        return self._lease(exclusive=False)

    def collect(self, roots: Callable[[ObjectLease], Iterable[StoredRef]]) -> int:
        """Sweep unreachable objects after reading all retained roots under the fence.

        The callback can read tree pages through the provided lease. It must
        enumerate every reachable object and abort on incomplete root evidence.
        """
        with self._lease(exclusive=True) as lease:
            from kernel_workspace_refs import prune_root_stages, registered_objects

            directory = lease._check()
            retained: dict[str, int] = {}
            for references in (registered_objects(lease), roots(lease)):
                for ref in references:
                    if ref.digest in retained and retained[ref.digest] == ref.size:
                        continue
                    _read_checked(directory, ref)
                    retained[ref.digest] = ref.size

            objects: list[str] = []
            stages: list[str] = []
            for name in os.listdir(directory):
                if re.fullmatch(r"\.incoming-[0-9a-f]{32}", name):
                    stages.append(name)
                    continue
                digest = name.removeprefix(_STAGE)
                if _DIGEST.fullmatch(digest):
                    if name != digest:
                        stages.append(name)
                    elif digest not in retained:
                        objects.append(name)
            # Remove uncommitted addresses before their staging markers. An
            # interrupted sweep must leave any surviving sync debt detectable.
            for name in objects:
                os.unlink(name, dir_fd=directory)
            if stages:
                os.fsync(directory)
            for name in stages:
                os.unlink(name, dir_fd=directory)
            if objects or stages:
                os.fsync(directory)
            prune_root_stages(lease)
            self._prune_index_stages(lease)
            return len(objects)

    @staticmethod
    def _prune_index_stages(lease: ObjectLease) -> None:
        with suppress(FileNotFoundError), _child_directory(lease._namespace, "index") as directory:
            _private_directory(directory)
            stages = [name for name in os.listdir(directory) if _INDEX_STAGE.fullmatch(name)]
            for name in stages:
                os.unlink(name, dir_fd=directory)
            if stages:
                os.fsync(directory)


def open_workspace_objects(
    parent: Path, namespace: str, *, enabled: bool = False,
) -> WorkspaceObjects | None:
    """Open a store only when the host's opt-in switch is enabled.

    The parent must already exist outside writable workcopies and checkpoint
    input enumeration. Namespace selection and retention policy belong to the host.
    """
    if enabled is not True:
        return None
    if not isinstance(namespace, str) or not _NAMESPACE.fullmatch(namespace):
        raise ValueError("invalid workspace object namespace")
    parent = Path(parent).absolute()
    return _open_namespace(parent, namespace, create=True)


def _open_namespace(parent: Path, namespace: str, *, create: bool) -> WorkspaceObjects:
    with _directory(parent) as parent_fd:
        _private_directory(parent_fd)
        with _child_directory(parent_fd, namespace, create=create) as root:
            identity = _private_directory(root)
            with _child_directory(root, "objects", create=create) as directory:
                objects_identity = _private_directory(directory)
            with _child_directory(root, "refs", create=create) as registry:
                refs_identity = _private_directory(registry)
            if create:
                # Also repair a previous initialization interrupted by a crash.
                os.fsync(root)
                os.fsync(parent_fd)
    return WorkspaceObjects(parent / namespace, identity, objects_identity, refs_identity)
