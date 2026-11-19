"""Optional, process-local Linux dirty-path hints for the workspace owner.

Under a controlled writer fence, call ``rebuild(writer_fenced=True)``, take a
full stat snapshot, then call ``poll(writer_fenced=True, after_full_sweep=True)``
under that same fence. At later boundaries call ``poll(writer_fenced=True)``.
Pass successful ``paths`` directly to WorkspaceIndex.refresh, including an empty
tuple. Paths are sorted workspace-relative byte strings. Refreshing a newly
watched directory path must include its descendants. Merge in explicit writer
footprints and any other invalidations required by the index's policy.

A sweep result means use the portable full stat path (``refresh(None)``), or
report unavailable if there is no safe observation. After coverage loss,
rebuild before the next baseline sweep. Close the feed if reconciliation fails,
if the coverage policy changes, or when relinquishing ownership. No feed state
is needed for checkpoints, resume, or portable fallback.

Only a serialized host owner should use this object. The caller's fence must
account for writers between polls, including write channels invisible to
inotify such as shared mmap, remote writers, and overmounts. Hardlink content
and link-count changes need an explicit footprint covering all affected aliases.
The host must also account for access changes in ancestors outside the workspace.
Polling gives dirty hints. The owner retains freshness authority and must
maintain the fence through reconciliation and any dependent use.
"""
from __future__ import annotations

import errno
import os
import struct
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass

_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ONLYDIR = 0x01000000
_IN_ISDIR = 0x40000000

_ARRIVAL = _IN_CREATE | _IN_MOVED_TO
_DEPARTURE = _IN_DELETE | _IN_MOVED_FROM
_INVALIDATED = _IN_DELETE_SELF | _IN_MOVE_SELF | _IN_UNMOUNT | _IN_IGNORED
_WATCH_EVENTS = (_IN_MODIFY | _IN_ATTRIB | _IN_CLOSE_WRITE | _ARRIVAL | _DEPARTURE
                 | _IN_DELETE_SELF | _IN_MOVE_SELF)
_EVENT = struct.Struct("=iIII")  # wd, mask, cookie, padded name length
_READ_BYTES = 64 * 1024
_MAX_READS = 128
_MAX_DEPTH = 256
_Path = str | bytes | os.PathLike[str] | os.PathLike[bytes]


@dataclass(frozen=True, slots=True)
class FeedUpdate:
    """A complete dirty footprint or a reason to discard hints and sweep."""

    paths: tuple[bytes, ...] = ()
    sweep_reason: str | None = None

    @property
    def requires_sweep(self) -> bool:
        return self.sweep_reason is not None


class _CoverageLost(Exception):
    pass


def _open_inotify() -> tuple[int, Callable[[int], int]]:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    init = libc.inotify_init1
    init.argtypes = (ctypes.c_int,)
    init.restype = ctypes.c_int
    add = libc.inotify_add_watch
    add.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32)
    add.restype = ctypes.c_int

    def checked(result: int) -> int:
        if result < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        return result

    fd = checked(init(os.O_NONBLOCK | os.O_CLOEXEC))

    def watch(directory_fd: int) -> int:
        # procfs resolves this link to the already opened directory. The traversal
        # that produced directory_fd used O_NOFOLLOW at each child component.
        path = f"/proc/self/fd/{directory_fd}".encode("ascii")
        return checked(add(fd, path, _WATCH_EVENTS | _IN_ONLYDIR))

    return fd, watch


def _records(data: bytes) -> Iterator[tuple[int, int, bytes]]:
    cursor = 0
    while cursor < len(data):
        if len(data) - cursor < _EVENT.size:
            raise _CoverageLost("malformed_event")
        wd, mask, _, size = _EVENT.unpack_from(data, cursor)
        cursor += _EVENT.size
        end = cursor + size
        if end > len(data):
            raise _CoverageLost("malformed_event")
        padded = data[cursor:end]
        name, separator, padding = padded.partition(b"\0")
        if (padded and (not separator or padding.strip(b"\0"))) or (
            name in (b".", b"..") or b"/" in name
        ):
            raise _CoverageLost("malformed_event")
        yield wd, mask, name
        cursor = end


class WorkspaceFeed:
    """Track an explicitly opted-in workspace with an immutable coverage policy.

    ``include_directory`` receives a relative byte path and can prune directory
    subtrees. The root is always watched. A matching full sweep must use the
    same coverage. The owner can still require a sweep for additional policies.
    """

    def __init__(
        self, root: _Path, *, enabled: bool = False,
        include_directory: Callable[[bytes], bool] | None = None,
    ) -> None:
        self._enabled = enabled
        self._root = os.path.abspath(os.fsencode(root)) if enabled else b""
        self._include_directory = include_directory
        self._fd = self._root_fd = -1
        self._owner_pid = 0
        self._reason = "closed" if enabled else "disabled"
        self._needs_sweep = True
        self._identity: tuple[int, int] | None = None
        self._add_watch: Callable[[int], int] | None = None
        self._watches: dict[int, bytes] = {}
        self._directories: set[bytes] = set()

    def close(self) -> None:
        """Close this process's descriptors. Safe to call again or after fork."""
        fd, root_fd = self._fd, self._root_fd
        self._fd = self._root_fd = -1
        self._owner_pid = 0
        self._reason = "closed" if self._enabled else "disabled"
        self._needs_sweep = True
        self._identity = None
        self._add_watch = None
        self._watches.clear()
        self._directories.clear()
        # Close the entire inotify instance rather than recycle watch descriptors
        # while older records bearing those numbers might still be queued.
        try:
            if fd >= 0:
                os.close(fd)
        finally:
            if root_fd >= 0:
                os.close(root_fd)

    def _lose(self, reason: str) -> FeedUpdate:
        self.close()
        self._reason = reason
        return FeedUpdate(sweep_reason=reason)

    def rebuild(self, *, writer_fenced: bool = False) -> bool:
        """Install a fresh watch generation before the owner's full stat sweep."""
        self.close()
        if not self._enabled:
            return False
        if sys.platform != "linux":
            self._lose("unsupported")
            return False
        if not writer_fenced:
            self._lose("unfenced")
            return False
        try:
            self._owner_pid = os.getpid()
            self._fd, self._add_watch = _open_inotify()
            self._root_fd = os.open(self._root, self._directory_flags())
            info = os.fstat(self._root_fd)
            self._identity = (info.st_dev, info.st_ino)
            self._watch_tree(self._root_fd, b"", 0)
            self._validate_root()
        except _CoverageLost as error:
            self._lose(str(error))
            return False
        except (OSError, AttributeError, ImportError):
            self._lose("watch_setup_failed")
            return False
        except BaseException:
            self.close()
            raise
        return True

    def poll(
        self, *, writer_fenced: bool = False, after_full_sweep: bool = False,
    ) -> FeedUpdate:
        """Drain one coverage generation at a caller-owned observation boundary.

        Set ``after_full_sweep`` only after successfully taking the stat baseline
        with this watch generation installed. The acknowledgement never drops
        queued events. Overflow, interruption or loss discards the whole batch.
        """
        if self._fd < 0:
            return FeedUpdate(sweep_reason=self._reason)
        if self._owner_pid != os.getpid():
            return self._lose("forked")
        if not writer_fenced:
            return self._lose("unfenced")
        if self._needs_sweep and not after_full_sweep:
            return FeedUpdate(sweep_reason="baseline_required")
        try:
            paths = self._drain()
            self._validate_root()
        except _CoverageLost as error:
            return self._lose(str(error))
        except OSError:
            return self._lose("drain_failed")
        except BaseException:
            self.close()
            raise
        self._needs_sweep = False
        return FeedUpdate(paths=paths)

    @staticmethod
    def _directory_flags() -> int:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    def _validate_root(self) -> None:
        try:
            info = os.stat(self._root, follow_symlinks=False)
        except OSError as error:
            raise _CoverageLost("root_changed") from error
        if (info.st_dev, info.st_ino) != self._identity:
            raise _CoverageLost("root_changed")

    def _included(self, path: bytes) -> bool:
        return self._include_directory is None or self._include_directory(path)

    def _watch_tree(self, fd: int, path: bytes, depth: int) -> None:
        if depth > _MAX_DEPTH:
            raise _CoverageLost("depth_limit")
        assert self._add_watch is not None
        wd = self._add_watch(fd)
        if wd in self._watches or path in self._directories:
            raise _CoverageLost("watch_reused")
        self._watches[wd] = path
        self._directories.add(path)
        prefix = path + b"/" if path else b""
        with os.scandir(fd) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    name = os.fsencode(entry.name)
                    child = prefix + name
                    if self._included(child):
                        child_fd = os.open(name, self._directory_flags(), dir_fd=fd)
                        try:
                            self._watch_tree(child_fd, child, depth + 1)
                        finally:
                            os.close(child_fd)

    def _watch_arrival(self, path: bytes) -> None:
        parts = path.split(b"/")
        if len(parts) > _MAX_DEPTH:
            raise _CoverageLost("depth_limit")
        fd = os.dup(self._root_fd)
        try:
            for name in parts:
                child_fd = os.open(name, self._directory_flags(), dir_fd=fd)
                os.close(fd)
                fd = child_fd
            self._watch_tree(fd, path, len(parts))
        except OSError as error:
            raise _CoverageLost("watch_setup_failed") from error
        finally:
            os.close(fd)

    def _record(
        self, wd: int, mask: int, name: bytes, dirty: set[bytes], arrivals: set[bytes],
    ) -> None:
        if mask & _IN_Q_OVERFLOW:
            raise _CoverageLost("overflow")
        if mask & _INVALIDATED:
            raise _CoverageLost("watch_lost")
        parent = self._watches.get(wd)
        if parent is None:
            raise _CoverageLost("unknown_watch")
        if not mask & _WATCH_EVENTS or mask & ~(_WATCH_EVENTS | _IN_ISDIR):
            raise _CoverageLost("unknown_event")
        if not name:
            if mask != _IN_ATTRIB and mask != (_IN_ATTRIB | _IN_ISDIR):
                raise _CoverageLost("malformed_event")
            if not parent:
                raise _CoverageLost("root_changed")
            dirty.add(parent)
            return
        path = parent + b"/" + name if parent else name
        if mask & _IN_ISDIR:
            if not self._included(path):
                return
            if mask & _DEPARTURE:
                # The remaining queue can reference the old location or its
                # descendants. Rebuild rather than guessing at watch ownership.
                raise _CoverageLost("directory_departed")
            if mask & _ARRIVAL:
                arrivals.add(path)
        dirty.add(path)

    def _drain(self) -> tuple[bytes, ...]:
        dirty: set[bytes] = set()
        arrivals: set[bytes] = set()
        for _ in range(_MAX_READS):
            try:
                data = os.read(self._fd, _READ_BYTES)
            except BlockingIOError as error:
                if error.errno != errno.EAGAIN:
                    raise
                if not arrivals:
                    return tuple(sorted(dirty))
                for path in sorted(arrivals):
                    self._watch_arrival(path)
                arrivals.clear()
                continue  # Include any events queued while installing watches.
            if not data:
                raise _CoverageLost("empty_read")
            for wd, mask, name in _records(data):
                self._record(wd, mask, name, dirty, arrivals)
        raise _CoverageLost("drain_limit")
