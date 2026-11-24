"""Safe working copy of the user's repository.

Every run works on a private copy under the run directory. The original files are only
written when the user passes --in-place, and then only files the run changed, and only if
the original still has the bytes it had when the run started (no clobbering of edits the
user made meanwhile).
"""
from __future__ import annotations

import difflib
import errno
import fnmatch
import hashlib
import os
import secrets
import shutil
import stat
import sys
import tempfile
import tokenize
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping

from . import _startup_optimization_enabled
from .engine.kernel_edit_merge import ConflictHunk, MergeApplied, MergeConflict, merge_texts
from .engine.kernel_secret_paths import SECRET_DIRS, is_secret_file

_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_DIRECTORY_FLAGS = _READ_FLAGS | os.O_DIRECTORY

# Bounds for the conflict report shown to the user (COLLAB-DRIFT piece F).
MAX_HUNKS_PER_FILE = 5
MAX_HUNK_LINES = 40
MAX_CONFLICT_FILES_SHOWN = 10

# Never copied: VCS metadata, environments, caches, build output.
IGNORE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
               ".tox", ".nox", ".venv", "venv", "env", "node_modules", "dist", "build",
               ".idea", ".vscode", ".eggs", "site-packages", ".stateframe"} | SECRET_DIRS
IGNORE_GLOBS = ("*.pyc", "*.pyo", "*.so", "*.dylib", "*.egg-info", ".DS_Store")
_LARGE_COPY_BYTES = 64 << 20
_MIN_RESERVE_BYTES = 1 << 30
_MAX_RESERVE_BYTES = 20 << 30
_TEXT_FILE_BYTES = 1 << 20


class WorkCopyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConflictReport:
    """One file whose run edits and outside edits changed the same lines. Line numbers are 1-based, inclusive."""

    path: str
    hunk_count: int
    hunks: tuple[dict[str, object], ...]  # bounded: first MAX_HUNKS_PER_FILE hunks, each side cut to MAX_HUNK_LINES
    spans: str  # one line for the refusal message, e.g. "run lines 2-2 vs original lines 2-2"


class MergeConflictError(WorkCopyError):
    """Outside and run edits overlap; nothing was written. ``conflicts`` carries the typed report."""

    def __init__(self, message: str, conflicts: tuple[ConflictReport, ...]) -> None:
        super().__init__(message)
        self.conflicts = conflicts


class WorkCopyScanError(WorkCopyError):
    pass


def _repository_scan_error(path: Path | str, error: OSError) -> WorkCopyScanError:
    detail = error.strerror or str(error)
    return WorkCopyScanError(f"cannot scan repository directory {path}: {detail}")


@dataclass
class Limits:
    max_file_bytes: int = 1024 * 1024


_Stamp = tuple[int, int, int, int, int, int, int, int, int]


def _stamp(info: os.stat_result) -> _Stamp:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink, info.st_uid, info.st_gid)


def _reflink(source: int, target: int) -> None:
    if sys.platform != "linux":
        raise OSError(errno.ENOTSUP, "file clone unavailable")
    import fcntl

    fcntl.ioctl(target, 0x40049409, source)  # FICLONE


class _SourceChanged(OSError):
    """The file got shorter while it was copied."""


class _Copies:
    """Copy regular files to independent inodes, including after a partial clone."""

    def __init__(self, clone: bool = True) -> None:
        self.clone, self.last_reflink = clone, False
        self.mode_only = False  # the copy of the user's repository holds new files: no times or attributes

    def file(self, source: int, target: Path, info: os.stat_result, *,
             data: bytes | None = None, writable: bool = False,
             before_write: Callable[[int], None] | None = None) -> _Stamp:
        self.last_reflink = False
        with target.open("xb") as output:
            if self.clone:
                try:
                    _reflink(source, output.fileno())
                except OSError:
                    output.seek(0)
                    output.truncate()
                    self.clone = False
                else:
                    if os.fstat(output.fileno()).st_size == info.st_size:
                        self.last_reflink = True
                        return self._metadata(source, output.fileno(), info, writable)
                    output.seek(0)
                    output.truncate()
                    self.clone = False
            if data is None:
                os.lseek(source, 0, os.SEEK_SET)
                # The source is checked again by the caller. Bound the copy if
                # a file grows after its admitted size was checked.
                remaining = info.st_size
                if before_write is not None:
                    before_write(remaining)
                while remaining:
                    block = os.read(source, min(remaining, 1 << 20))
                    if not block:
                        raise _SourceChanged(errno.EIO, "snapshot source became shorter")
                    output.write(block)
                    remaining -= len(block)
            else:
                if before_write is not None:
                    before_write(len(data))
                output.write(data)
            output.flush()
            return self._metadata(source, output.fileno(), info, writable)

    def _metadata(self, source: int, fd: int, info: os.stat_result, writable: bool) -> _Stamp:
        mode = stat.S_IMODE(info.st_mode)
        if self.mode_only:
            os.fchmod(fd, mode)
        elif sys.platform == "linux":
            # Use open inodes, including for extended attributes and ACLs.
            shutil.copystat(source, fd)
            if writable and not mode & stat.S_IWUSR:
                os.fchmod(fd, mode | stat.S_IWUSR)
        else:
            os.utime(fd, ns=(info.st_atime_ns, info.st_mtime_ns))
            os.fchmod(fd, mode | stat.S_IWUSR if writable else mode)
        return _stamp(os.fstat(fd))


def _free_space(destination: Path) -> tuple[int | None, int]:
    """Free bytes where the copy goes, and how many a large copy must leave for everything else."""
    where = destination.parent
    while not os.path.lexists(where) and where != where.parent:  # the run directory is made later
        where = where.parent
    try:
        space = os.statvfs(where)
    except (AttributeError, OSError):
        return None, 0
    total = getattr(space, "f_blocks", 0) * space.f_frsize
    return space.f_bavail * space.f_frsize, min(max(_MIN_RESERVE_BYTES, total // 20), _MAX_RESERVE_BYTES)


def _copy_bytes(original: Path, max_file_bytes: int) -> int:
    """Bytes the copy would take. A part that cannot be listed is left to the copy, which reports it."""
    needed = 0
    for directory, dirnames, filenames in os.walk(original, followlinks=False):
        here = Path(directory)
        dirnames[:] = [name for name in dirnames
                       if not _copy_ignored(name, True) and not (here / name).is_symlink()]
        for name in filenames:
            if _copy_ignored(name, False):
                continue
            try:
                info = (here / name).lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                needed += info.st_size
    return needed


def _probe_reflink(original: Path, destination: Path) -> bool:
    """Whether files can be cloned here. Uses its own file; a repository file is never opened for this."""
    where = destination.parent
    while not os.path.lexists(where) and where != where.parent:
        where = where.parent
    try:
        if original.stat().st_dev != where.stat().st_dev:
            return False  # no promise about space; the copy itself still tries to clone
        with tempfile.NamedTemporaryFile(dir=where) as source, tempfile.NamedTemporaryFile(dir=where) as target:
            source.write(b"x" * 4096)
            source.flush()
            _reflink(source.fileno(), target.fileno())
            return os.fstat(target.fileno()).st_size == 4096
    except OSError:
        return False


def _check_copy_space(original: Path, destination: Path, max_file_bytes: int,
                      error: OSError | None = None, *, copies_required: int = 2) -> bool:
    """Refuse a start whose copies do not fit; a large repository must also leave a reserve."""
    free, reserve = _free_space(destination)
    cloned = error is None and _probe_reflink(original, destination)
    if error is None and free is None:
        return cloned
    size = _copy_bytes(original, max_file_bytes)
    needed = 0 if cloned else copies_required * size
    if size <= _LARGE_COPY_BYTES:
        reserve = 0
    if error is not None or (free is not None and needed + reserve > free):
        detail = f" ({error.strerror or error})" if error is not None else ""
        available = "unknown" if free is None else str(free)
        kept = f" and {reserve} bytes to stay free" if reserve else ""
        raise WorkCopyError(
            f"repository copy needs {needed} bytes{kept}; {available} bytes are free{detail}; "
            "choose a smaller tree with --repo")
    return cloned


class _CopySpace:
    """Looks at the disk again while a large copy grows: the repository can grow meanwhile."""

    def __init__(self, destination: Path) -> None:
        self.destination, self.written, self.checked = destination, 0, 0

    def add(self, count: int) -> None:
        self.written += count
        if self.written <= _LARGE_COPY_BYTES or self.written - self.checked < _LARGE_COPY_BYTES:
            return
        self.checked = self.written
        free, reserve = _free_space(self.destination)
        if free is not None and free < reserve:
            raise OSError(errno.ENOSPC, "the disk is nearly full")


def _cleanup_partial_copy(paths: tuple[Path, ...], error: OSError) -> None:
    for path in paths:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError as cleanup_error:
            raise WorkCopyError(f"partial working copy cleanup failed: {cleanup_error}") from error


def _check_apply_space(destination: Path, changes: Mapping[str, bytes | None]) -> None:
    needed = sum(len(data) for data in changes.values() if data is not None)
    free, _reserve = _free_space(destination / "file")
    if free is not None and needed > free:
        raise WorkCopyError(f"applying the result needs {needed} bytes; {free} bytes are free")


def _raise_disk_full(original: Path, destination: Path, max_file_bytes: int,
                     paths: tuple[Path, ...], error: OSError, *, copies_required: int = 2) -> None:
    codes = {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}
    if isinstance(error, shutil.Error):  # copytree reports its failures as text
        failures = error.args[0] if error.args and isinstance(error.args[0], list) else []
        text = " ".join(str(failure[-1]) for failure in failures)
        code = next((code for code in codes if f"[Errno {code}]" in text), None)
        if code is None:
            return
        error = OSError(code, os.strerror(code))
    if error.errno not in codes:
        return
    _cleanup_partial_copy(paths, error)
    _check_copy_space(original, destination, max_file_bytes, error,
                      copies_required=copies_required)


@dataclass
class WorkCopy:
    original: Path
    copy: Path
    baseline: dict[str, str] = field(default_factory=dict)  # relative path -> sha256 at start
    skipped: list[str] = field(default_factory=list)
    snapshot: Path | None = None  # immutable copy of the start state (diff baseline), if the run made one
    snapshot_reader: Callable[[str], bytes] | None = field(default=None, repr=False)
    skip_reasons: dict[str, str] = field(default_factory=dict)
    large_files: set[str] = field(default_factory=set)
    _copies: _Copies | None = field(default=None, init=False, repr=False)
    _admission: object | None = field(default=None, init=False, repr=False, compare=False)

    def note_skipped(self, rel: str, reason: str) -> None:
        self.skipped.append(rel)
        self.skip_reasons[rel] = reason

    def text_files(self, wanted: Callable[[str], bool] | None = None,
                   large: Callable[[str], bool] | None = None) -> dict[str, str]:
        """Text files in the copy, keyed by POSIX relative path; only ``wanted`` ones are read.

        An unchanged file over 1 MiB is read only when ``large`` selects it.
        """
        out: dict[str, str] = {}
        for rel in sorted(self.baseline):
            path = self.copy / rel
            if wanted is not None and path.suffix.lower() not in {".py", ".pyi"} and not wanted(rel):
                continue
            if rel in self.large_files and not (large is not None and large(rel)):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    if _hash_file(path) == self.baseline[rel]:
                        continue
                except OSError as error:
                    raise WorkCopyError(f"cannot read working copy file {rel}") from error
            try:
                if path.suffix.lower() in {".py", ".pyi"}:
                    with tokenize.open(path) as source:
                        out[rel] = source.read()
                else:
                    out[rel] = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError, SyntaxError) as error:
                if path.suffix.lower() in {".py", ".pyi"}:
                    raise WorkCopyError(
                        f"cannot decode or read Python source file {rel} in the working copy"
                    ) from error
                continue
        return out


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_copy_storage_full(error: OSError) -> bool:
    return error.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path, copies: _Copies | None = None, *,
              work: WorkCopy | None = None, space: _CopySpace | None = None,
              preserve_symlinks: bool = False,
              preserve_directory_modes: bool = True) -> Path:
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if os.path.lexists(destination):
        raise WorkCopyError(f"working copy destination already exists: {destination}")
    destination.mkdir(parents=True)
    copier = copies or _Copies(False)
    directories = []

    def scan_error(error: OSError) -> None:
        path = error.filename or source
        raise _repository_scan_error(path, error) from error

    def note_skip(path: Path, reason: str) -> None:
        if work is not None:
            work.note_skipped(path.relative_to(source).as_posix(), reason)

    for directory, dirnames, filenames in os.walk(source, followlinks=False, onerror=scan_error):
        here = Path(directory)
        relative_dir = here.relative_to(source)
        target_dir = destination / relative_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        directories.append((here, target_dir))
        kept_dirs = []
        for name in sorted(dirnames):
            path = here / name
            try:
                info = path.lstat()
            except OSError as error:
                raise _repository_scan_error(path, error) from error
            if stat.S_ISLNK(info.st_mode):
                if preserve_symlinks:
                    (target_dir / name).symlink_to(os.readlink(path), target_is_directory=True)
                continue
            kept_dirs.append(name)
        dirnames[:] = kept_dirs
        for name in sorted(filenames):
            path = here / name
            rel = path.relative_to(source).as_posix()
            try:
                before = path.lstat()
            except OSError as error:
                raise WorkCopyScanError(f"cannot inspect repository file {path}: {error}") from error
            if stat.S_ISLNK(before.st_mode):
                if preserve_symlinks:
                    (target_dir / name).symlink_to(os.readlink(path))
                else:
                    note_skip(path, "symbolic link")
                continue
            if not stat.S_ISREG(before.st_mode):
                note_skip(path, "special file")
                continue
            try:
                source_fd = os.open(path, _READ_FLAGS)
            except OSError as error:
                raise WorkCopyScanError(f"cannot read repository file {path}: {error}") from error
            with os.fdopen(source_fd, "rb"):
                info = os.fstat(source_fd)
                if not stat.S_ISREG(info.st_mode):
                    note_skip(path, "special file")
                    continue
                if (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
                    raise WorkCopyError(f"repository file changed before copy: {rel}")
                if work is not None and info.st_size > _TEXT_FILE_BYTES:
                    work.large_files.add(rel)
                target = target_dir / name
                try:
                    copier.file(source_fd, target, info,
                                before_write=space.add if space is not None else None)
                except OSError as error:
                    if error.errno in {errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)}:
                        raise
                    raise WorkCopyScanError(f"cannot copy repository file {path}: {error}") from error
                if _stamp(os.fstat(source_fd)) != _stamp(info):
                    raise WorkCopyError(f"copy source changed during copy: {path}")
                if work is not None:
                    work.baseline[rel] = _hash_file(target)
    if preserve_directory_modes:
        for path, target in reversed(directories):
            shutil.copystat(path, target, follow_symlinks=False)
    return destination


def copy_tree_checked(source: Path, destination: Path, *, work: WorkCopy | None = None,
                      max_file_bytes: int = 1 << 20, copies_required: int = 1,
                      preserve_symlinks: bool = False,
                      preserve_directory_modes: bool = True) -> Path:
    # A run copying its own trees keeps files the first copy would leave out.
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if not source.is_dir():
        raise WorkCopyError(f"copy source is not a directory: {source}")
    if os.path.lexists(destination):
        raise WorkCopyError(f"working copy destination already exists: {destination}")
    _check_copy_space(source, destination, max_file_bytes, copies_required=copies_required)
    copies = _Copies()
    if work is not None:
        work._copies = copies
    try:
        return copy_tree(source, destination, copies, work=work,
                         space=_CopySpace(destination), preserve_symlinks=preserve_symlinks,
                         preserve_directory_modes=preserve_directory_modes)
    except OSError as error:
        _raise_disk_full(source, destination, max_file_bytes, (destination,), error,
                         copies_required=copies_required)
        raise


def _ignored(name: str, is_dir: bool) -> bool:
    if not name.isascii():
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            return True  # not valid UTF-8: no request, journal or patch can name it
    if is_dir:
        return name in IGNORE_DIRS or name.casefold() in SECRET_DIRS or name.endswith(".egg-info")
    return is_secret_file(name) or any(fnmatch.fnmatch(name, pattern) for pattern in IGNORE_GLOBS)


_copy_ignored = _ignored


def frame_source_path_allowed(path: str) -> bool:
    """Use the copy's exclusion rules for source added after the copy was made."""
    parts = path.split("/") if isinstance(path, str) else []
    return bool(parts and all(part not in {"", ".", ".."} for part in parts)
                and not any(_ignored(part, True) for part in parts[:-1])
                and not _ignored(parts[-1], False))


def _walk_source(
    root: Path, onerror: Callable[[Path, OSError], None],
) -> Iterator[tuple[Path, list[str], int]]:
    """Walk in legacy order while retaining directory descriptors and error paths."""
    # One close marker per open ancestor keeps the descriptor lifetime bounded by depth.
    pending: list[tuple[Path | None, int | None]] = [(root, None)]
    opened: list[int] = []
    try:
        while pending:
            path, parent_fd = pending.pop()
            if path is None:
                os.close(opened.pop())
                continue
            name = path if parent_fd is None else path.name
            try:
                fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            except OSError as error:
                if parent_fd is not None and error.errno in {errno.ENOTDIR, errno.ELOOP}:
                    # walk does not descend into a directory symlink, including one
                    # installed after enumeration. Inspect the entry without following it.
                    try:
                        if stat.S_ISLNK(os.stat(name, dir_fd=parent_fd, follow_symlinks=False).st_mode):
                            continue
                    except OSError as inspect_error:
                        error = inspect_error
                onerror(path, error)
                continue
            opened.append(fd)
            pending.append((None, None))
            children, files = [], []
            try:
                with os.scandir(fd) as entries:
                    for entry in entries:
                        try:
                            is_dir = entry.is_dir()
                        except OSError:
                            is_dir = False  # Match os.walk's classification of symlink loops.
                        if is_dir:
                            children.append(entry.name)
                        else:
                            files.append(entry.name)
            except OSError as error:
                onerror(path, error)
                continue
            children = sorted(name for name in children if not _ignored(name, True))
            yield path, sorted(files), fd
            pending.extend((path / name, fd) for name in reversed(children))
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _source_file_error(operation: str, path: Path, error: OSError) -> WorkCopyScanError:
    if error.filename is not None:
        # Descriptor-relative syscalls report only the entry name.
        error = OSError(error.errno, error.strerror, os.fspath(path))
    return WorkCopyScanError(f"cannot {operation} repository file {path}: {error}")


def _copy_start_file(copies: _Copies, source: int, target: Path, rel: str, space: _CopySpace) -> bool:
    """Copy one open repository file; retry once if it changes meanwhile.

    False means a file over 1 MiB, such as a growing log, kept changing and is left out.
    """
    large = False
    for _attempt in range(2):
        info = os.fstat(source)
        large = large or info.st_size > _TEXT_FILE_BYTES
        try:
            copies.file(source, target, info, before_write=space.add)
            if _stamp(os.fstat(source)) == _stamp(info):
                return True
        except _SourceChanged:
            pass
        target.unlink()
    if large:
        return False
    raise WorkCopyError(f"repository file changed during copy: {rel}")


def create(original: Path, destination: Path, limits: Limits | None = None, *,
           admit_snapshot: bool | None = None) -> WorkCopy:
    """Copy ``original`` into ``destination`` (which must not exist), recording hashes."""
    limits = limits or Limits()
    original = original.resolve()
    if not original.is_dir():
        raise WorkCopyError(f"--repo is not a directory: {original}")
    destination = destination.resolve()
    if destination.exists():
        raise WorkCopyError(f"working copy destination already exists: {destination}")
    if destination.is_relative_to(original):
        raise WorkCopyError("run directory must be outside the repository")
    _check_copy_space(original, destination, limits.max_file_bytes)
    work = WorkCopy(original=original, copy=destination)
    copies = _Copies()
    work._copies = copies
    if admit_snapshot is None:
        admit_snapshot = _startup_optimization_enabled()
    if admit_snapshot:
        from .workspace_start_snapshot import admit_workcopy

        try:
            admit_workcopy(work, limits)
        except OSError as error:
            _raise_disk_full(original, destination, limits.max_file_bytes,
                             (destination,), error)
            raise
        return work
    space = _CopySpace(destination)

    def scan_error(path: Path, error: OSError) -> None:
        raise _repository_scan_error(path, error) from error

    copies.mode_only = True
    try:
        destination.mkdir(parents=True)
        for directory, filenames, dir_fd in _walk_source(original, onerror=scan_error):
            here = directory.relative_to(original)
            for name in filenames:
                relative = here / name
                rel = relative.as_posix()
                if _ignored(name, False):
                    work.note_skipped(rel, "excluded by file name")
                    continue
                source = directory / name
                try:
                    before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError as error:
                    raise _source_file_error("inspect", source, error) from error
                if not stat.S_ISREG(before.st_mode):
                    work.note_skipped(rel, "symbolic link" if stat.S_ISLNK(before.st_mode) else "special file")
                    continue
                try:
                    descriptor = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
                except OSError as error:
                    if error.errno in {errno.ELOOP, errno.ENXIO}:
                        work.note_skipped(rel, "symbolic link" if error.errno == errno.ELOOP else "special file")
                        continue
                    if _is_copy_storage_full(error):
                        raise
                    raise _source_file_error("read", source, error) from error
                with os.fdopen(descriptor, "rb"):
                    info = os.fstat(descriptor)
                    # An editor's atomic save may have replaced the file since the listing: copy the new one.
                    if not stat.S_ISREG(info.st_mode):
                        work.note_skipped(rel, "special file")
                        continue
                    target = destination / relative
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        copied = _copy_start_file(copies, descriptor, target, rel, space)
                    except OSError as error:
                        if _is_copy_storage_full(error):
                            raise
                        raise _source_file_error("copy", source, error) from error
                    if not copied:
                        work.note_skipped(rel, "changed while copying")
                        continue
                work.baseline[rel] = _hash_file(target)
                if target.stat().st_size > _TEXT_FILE_BYTES:
                    work.large_files.add(rel)
    except OSError as error:
        _raise_disk_full(original, destination, limits.max_file_bytes, (destination,), error)
        raise
    finally:
        copies.mode_only = False
    return work


def changed_files(work: WorkCopy) -> dict[str, str]:
    """relative path -> 'modified' | 'added' | 'deleted' in the copy versus the baseline."""
    changes: dict[str, str] = {}
    current: dict[str, str] = {}

    def scan_error(error: OSError) -> None:
        path = error.filename or work.copy
        detail = error.strerror or str(error)
        raise WorkCopyScanError(f"cannot scan working copy directory {path}: {detail}") from error

    for directory, dirnames, filenames in os.walk(work.copy, followlinks=False, onerror=scan_error):
        here = Path(directory)
        dirnames[:] = sorted(name for name in dirnames
                             if not _ignored(name, True) and not (here / name).is_symlink())
        for name in sorted(filenames):
            path = here / name
            if path.is_symlink() or _ignored(name, False):
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(work.copy).as_posix()
            try:
                current[rel] = _hash_file(path)
            except OSError as error:
                raise WorkCopyScanError(f"cannot read working copy file {path}: {error}") from error
    for rel, digest in current.items():
        if rel not in work.baseline:
            changes[rel] = "added"
        elif work.baseline[rel] != digest:
            changes[rel] = "modified"
    for rel in work.baseline:
        if rel not in current:
            changes[rel] = "deleted"
    return dict(sorted(changes.items()))


def changes_from_digests(baseline: Mapping[str, str], digests: Mapping[str, str]) -> dict[str, str]:
    changes = {path: "deleted" for path in baseline.keys() - digests.keys()}
    changes.update({path: "added" if path not in baseline else "modified" for path, digest in digests.items()
                    if path not in baseline or baseline[path] != digest})
    return dict(sorted(changes.items()))


def unified_diff(work: WorkCopy, *, scan_bytes: Callable[[bytes], object] | None = None) -> str:
    """Diff exact changed bytes, scanning paths and both inputs before formatting when requested."""
    chunks = []
    for rel, kind in changed_files(work).items():
        if scan_bytes is not None:
            scan_bytes(os.fsencode(rel))
        before_bytes = _read_original_bytes(work, rel) if kind != "added" else b""
        after_bytes = (work.copy / rel).read_bytes() if kind != "deleted" else b""
        if scan_bytes is not None:
            scan_bytes(before_bytes)
            scan_bytes(after_bytes)
        before = _diff_text(before_bytes)
        after = _diff_text(after_bytes)
        if before is None or after is None:
            chunks.append(
                f"binary/line-ending change: {rel} "
                "(text diff omitted; byte-level change recorded).\n"
            )
            continue
        file_diff = list(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{rel}" if kind != "added" else "/dev/null",
            tofile=f"b/{rel}" if kind != "deleted" else "/dev/null"))
        for index, line in enumerate(file_diff):
            if line.startswith((" ", "+", "-")) and not line.endswith(("\n", "\r")):
                file_diff[index] = line + "\n\\ No newline at end of file\n"
        if not file_diff:
            chunks.append(
                f"binary/line-ending change: {rel} "
                "(text diff omitted; byte-level change recorded).\n"
            )
            continue
        chunks.extend(file_diff)
    return "".join(chunks)


def _diff_text(data: bytes) -> str | None:
    """Return displayable UTF-8 text, preserving physical line endings."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _read_original_bytes(work: WorkCopy, rel: str) -> bytes:
    """Baseline bytes from the immutable start snapshot, or the unchanged original."""
    if work.snapshot_reader is not None:
        try:
            data = work.snapshot_reader(rel)
        except (OSError, ValueError) as error:
            raise WorkCopyError(f"original {rel} snapshot is unavailable") from error
    else:
        path = (work.snapshot if work.snapshot is not None else work.original) / rel
        data = path.read_bytes() if path.is_file() else b""
    if _sha(data) != work.baseline.get(rel):
        raise WorkCopyError(f"original {rel} changed during the run; diff would be wrong")
    return data


def _swap_deletions(files: dict[str, bytes | None]) -> set[str]:
    """Deletions that must come first: a file above or below a path that is written."""
    written = {rel for rel, data in files.items() if data is not None}
    above = {parent.as_posix() for rel in written for parent in Path(rel).parents}
    return {rel for rel, data in files.items() if data is None
            and (rel in above or any(parent.as_posix() in written for parent in Path(rel).parents))}


def _remove_empty_directories(root: Path, rel: str) -> None:
    """Remove ``rel`` if it is a tree of empty directories; anything else stays for the apply to refuse."""
    try:
        with _parent_fd(root, rel) as parent:
            _rmdir_tree(parent, Path(rel).name)
    except OSError:
        return


def _rmdir_tree(parent: int, name: str) -> None:
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        for child in os.listdir(fd):
            _rmdir_tree(fd, child)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent)


def apply_files(work: WorkCopy, files: dict[str, bytes | None], *, declared: set[str]) -> list[str]:
    """Apply the sealed candidate's writes and deletions to the copy.

    Only paths the run declared to the engine (or new files under the copy) are written,
    each must stay inside the copy, and symlinks are refused.
    """
    blockers = _swap_deletions(files) & declared
    if blockers:
        # A file became a directory or the reverse: the copy is private, so clear the old side first.
        done = {rel for rel in blockers if (work.copy / rel).is_dir() and not (work.copy / rel).is_symlink()}
        applied = apply_files(work, dict.fromkeys(blockers - done), declared=declared)
        rest = {rel: data for rel, data in files.items() if rel not in blockers}
        for rel, data in rest.items():
            if data is not None and (rel in declared or rel not in work.baseline):
                _remove_empty_directories(work.copy, rel)
        return applied + apply_files(work, rest, declared=declared)
    changes: dict[str, bytes | None] = {}
    baseline: dict[str, str] = {}
    for rel, data in sorted(files.items()):
        _relative_parts(rel)
        if rel not in declared and rel in work.baseline:
            continue
        # Test/tool litter (caches, bytecode, junk) is never product output, even if the engine captured it.
        if any(_ignored(part, True) for part in Path(rel).parts[:-1]) or _ignored(Path(rel).name, False):
            continue
        try:
            with _parent_fd(work.copy, rel) as parent:
                current, _ = _read_at(parent, Path(rel).name)
        except FileNotFoundError:
            current = None
        except NotADirectoryError as error:
            if data is not None:
                raise WorkCopyError(f"cannot read candidate destination {rel}: {error}") from error
            current = None  # an ancestor is a file now, so the path is already gone
        except OSError as error:
            raise WorkCopyError(f"cannot read candidate destination {rel}: {error}") from error
        if current == data:
            continue
        if current is not None:
            baseline[rel] = _sha(current)
        changes[rel] = data
    return guarded_apply(work.copy, changes, baseline, what="candidate apply")


def _relative_parts(rel: str) -> tuple[str, ...]:
    path = Path(rel)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise WorkCopyError(f"path escapes the repository: {rel}")
    return path.parts


@contextmanager
def _parent_fd(root: Path, rel: str, *, create: bool = False,
               created_dirs: list[tuple[str, tuple[int, int], tuple[int, int]]] | None = None):
    """Walk every component without resolving links; anchor writes to an open directory."""
    parts = _relative_parts(rel)
    root = root.absolute()  # absolute is lexical; resolve would hide symlinks
    if ".." in root.parts:
        raise WorkCopyError(f"parent traversal in repository root: {root}")
    descriptor = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        root_parts = root.parts[1:]
        for index, part in enumerate((*root_parts, *parts[:-1])):
            made = False
            parent_identity = None
            try:
                info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise
                parent_stat = os.fstat(descriptor)
                parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
                try:
                    os.mkdir(part, dir_fd=descriptor)
                    made = True
                except FileExistsError:
                    pass  # the no-follow open below validates a concurrent creation
            else:
                if stat.S_ISLNK(info.st_mode):
                    raise WorkCopyError(f"a parent directory is a symlink: {rel}")
            if made and created_dirs is not None and index >= len(root_parts):
                child_stat = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISDIR(child_stat.st_mode):
                    rel_dir = "/".join(parts[:index - len(root_parts) + 1])
                    created_dirs.append((rel_dir, parent_identity,
                                         (child_stat.st_dev, child_stat.st_ino)))
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def _stat_destination(parent: int, name: str) -> os.stat_result | None:
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise WorkCopyError(f"destination is a symlink: {name}")
    if not stat.S_ISREG(info.st_mode):
        raise WorkCopyError(f"destination is not a regular file: {name}")
    return info


def _read_at(parent: int, name: str) -> tuple[bytes | None, int | None]:
    """Read only regular files, refusing dangling links and races at the open itself."""
    if _stat_destination(parent, name) is None:
        return None, None
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(descriptor, "rb") as source:
        opened = os.fstat(source.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise WorkCopyError(f"destination is not a regular file: {name}")
        return source.read(), stat.S_IMODE(opened.st_mode)


def _check_bytes(current: bytes | None, baseline_sha: str | None) -> str | None:
    if baseline_sha is None:
        return "was created in the original meanwhile" if current is not None else None
    if current is None or _sha(current) != baseline_sha:
        return "changed in the original since the start"
    return None


def _apply_conflict(current: bytes | None, mode: int | None, baseline_sha: str | None,
                    values: list[dict[str, str | int] | None] | None) -> str | None:
    """Why the observed destination conflicts with its baseline or accepted retry state."""
    why = _check_bytes(current, baseline_sha)
    if current is None:
        return why if why and None not in (values or ()) else None
    digest = _sha(current)
    for value in values or ():
        if value is not None and value["sha256"] == digest:
            if "mode" not in value or value["mode"] == mode:
                return None
            return why or "changed in the original since the start"
    return why


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _verify_parent_identity(original: Path, rel: str, parent: int,
                            expected: tuple[int, int] | None = None) -> None:
    """Refuse a parent name that no longer identifies the directory observed earlier."""
    with _parent_fd(original, rel) as named_parent:
        opened = _identity(os.fstat(parent))
        named = _identity(os.fstat(named_parent))
        if opened != named or (expected is not None and opened != expected):
            raise WorkCopyError(f"parent directory changed while applying: {rel}")


def _verify_staged_identity(parent: int, name: str, expected: tuple[int, int]) -> None:
    """Refuse a staging name that no longer identifies the file this apply created."""
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(named.st_mode) or _identity(named) != expected:
        raise WorkCopyError("staged temporary changed while applying")


def _unlink_staged(parent: int, name: str, expected: tuple[int, int]) -> None:
    """Remove a staging name only while it still names the file this apply created."""
    try:
        named = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISREG(named.st_mode) and _identity(named) == expected:
            os.unlink(name, dir_fd=parent)
    except FileNotFoundError:
        pass


@contextmanager
def _apply_parent_fd(
    original: Path,
    rel: str,
    staged: dict[str, tuple[tuple[int, int], str, tuple[int, int]]],
    *,
    created_dirs: list[tuple[str, tuple[int, int], tuple[int, int]]] | None = None,
):
    with _parent_fd(original, rel, create=created_dirs is not None, created_dirs=created_dirs) as parent:
        try:
            yield parent
        except (OSError, WorkCopyError):
            # The live descriptor can still reach a parent moved outside the repository.
            with suppress(OSError):
                identity = _identity(os.fstat(parent))
                for parent_identity, temp, temp_identity in staged.values():
                    if parent_identity == identity:
                        with suppress(OSError):
                            _unlink_staged(parent, temp, temp_identity)
            raise


def _recover_staged_stat(descriptor: int) -> os.stat_result:
    """Recover identity through the open descriptor when fstat fails."""
    try:
        opened = os.stat(f"/proc/self/fd/{descriptor}")
    except OSError:
        opened = os.stat(f"/dev/fd/{descriptor}")
    if not stat.S_ISREG(opened.st_mode):
        raise OSError("staged temporary is not a regular file")
    return opened


def _new_file_mode(parent: int) -> int:
    """Measure this directory's creation mode without changing the process umask."""
    # This probe is always empty. A data-bearing temp is created separately at 0600.
    name = f".stateframe-{secrets.token_hex(16)}.mode"
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o666, dir_fd=parent)
    try:
        try:
            info = os.fstat(descriptor)
        except OSError:
            with suppress(OSError):
                _unlink_staged(parent, name, _identity(_recover_staged_stat(descriptor)))
            raise
        _unlink_staged(parent, name, _identity(info))
        return stat.S_IMODE(info.st_mode)
    finally:
        os.close(descriptor)


def _cleanup_created_dirs(
    original: Path,
    created_dirs: list[tuple[str, tuple[int, int], tuple[int, int]]],
) -> None:
    """Remove only the same empty directories this apply created, deepest first."""
    for rel, expected_parent, expected_child in reversed(created_dirs):
        try:
            with _parent_fd(original, rel) as parent:
                if _identity(os.fstat(parent)) != expected_parent:
                    continue
                name = Path(rel).name
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISDIR(named.st_mode) or _identity(named) != expected_child:
                    continue
                os.rmdir(name, dir_fd=parent)
        except (FileNotFoundError, NotADirectoryError, WorkCopyError, OSError):
            # Cleanup is best effort: a non-empty directory or a changed path is preserved.
            pass


def _remove_staged_temps(
    original: Path,
    staged: dict[str, tuple[tuple[int, int], str, tuple[int, int]]],
) -> None:
    """Remove our staged names, including when their parent was renamed inside the tree."""
    displaced: dict[tuple[int, int], list[tuple[str, tuple[int, int]]]] = {}
    for rel, (parent_identity, temp, temp_identity) in staged.items():
        try:
            with _parent_fd(original, rel) as parent:
                if _identity(os.fstat(parent)) == parent_identity:
                    _unlink_staged(parent, temp, temp_identity)
                else:
                    displaced.setdefault(parent_identity, []).append((temp, temp_identity))
        except (FileNotFoundError, NotADirectoryError, WorkCopyError, OSError):
            displaced.setdefault(parent_identity, []).append((temp, temp_identity))
    if not displaced:
        return

    # A parent may have been moved to another name while a replacement was in flight.
    # Locate it by inode under the repository root, without following any symlinks, so
    # its temp can still be removed without retaining one descriptor per changed file.
    pending = dict(displaced)
    stack = [""]
    visited: set[tuple[int, int]] = set()
    while stack and pending:
        rel_dir = stack.pop()
        probe = f"{rel_dir}/.stateframe-cleanup" if rel_dir else ".stateframe-cleanup"
        try:
            with _parent_fd(original, probe) as parent:
                identity = _identity(os.fstat(parent))
                if identity in visited:
                    continue
                visited.add(identity)
                for temp, temp_identity in pending.pop(identity, []):
                    with suppress(OSError):
                        _unlink_staged(parent, temp, temp_identity)
                if not pending:
                    break
                try:
                    names = os.listdir(parent)
                except OSError:
                    continue
                for name in names:
                    try:
                        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    except OSError:
                        continue
                    if not stat.S_ISDIR(info.st_mode):
                        continue
                    child_identity = _identity(info)
                    child_rel = f"{rel_dir}/{name}" if rel_dir else name
                    if child_identity in pending:
                        try:
                            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                            dir_fd=parent)
                        except OSError:
                            continue
                        try:
                            if _identity(os.fstat(child)) == child_identity:
                                for temp, temp_identity in pending.pop(child_identity):
                                    with suppress(OSError):
                                        _unlink_staged(child, temp, temp_identity)
                        finally:
                            os.close(child)
                    if pending:
                        stack.append(child_rel)
        except (OSError, WorkCopyError):
            continue


def guarded_apply(
    original: Path,
    changes: dict[str, bytes | None],
    baseline: dict[str, str],
    *,
    what: str,
    alternatives: dict[str, list[dict[str, str | int] | None]] | None = None,
) -> list[str]:
    """Write ``changes`` (rel -> new bytes, or None = delete) into ``original``.

    Every path is validated against ``baseline`` (sha at start; absent = new file) before staging and
    again as a set before the first replacement; each is re-checked before its own write. Writes are
    atomic per file (temp + rename), with temporary contents staged beside destinations. Parent names
    are reopened without following links before mutation; writes stay anchored to the
    directory descriptors reopened one file at a time after their identities are checked.
    The whole set is not atomic: on an I/O error the files already written stay written and the error
    names them; nothing is rolled back destructively.
    """
    # Preflight may defer a retry match; only the write pass can authorize it.
    conflicts = []
    destination_modes: dict[str, int | None] = {}
    for rel in sorted(changes):
        try:
            with _parent_fd(original, rel) as parent:
                current, mode = _read_at(parent, Path(rel).name)
        except FileNotFoundError:
            current, mode = None, None
        except (OSError, WorkCopyError) as error:
            conflicts.append(f"{rel} ({error})")
            continue
        if why := _apply_conflict(current, mode, baseline.get(rel), (alternatives or {}).get(rel)):
            conflicts.append(f"{rel} ({why})")
        if changes[rel] is not None:
            destination_modes[rel] = mode
    if conflicts:
        raise WorkCopyError(f"refusing {what}: " + "; ".join(conflicts[:10]))
    # Keep identities and names only.  Holding a parent and temp descriptor for every
    # changed file makes large promotions exhaust the process descriptor limit.
    staged: dict[str, tuple[tuple[int, int], str, tuple[int, int]]] = {}
    checked: dict[str, tuple[int, int]] = {}
    created_dirs: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    applied: list[str] = []
    succeeded = False
    try:
        for rel, data in sorted(changes.items()):
            if data is None:
                continue
            try:
                with _apply_parent_fd(original, rel, staged, created_dirs=created_dirs) as parent:
                    parent_identity = _identity(os.fstat(parent))
                    destination_mode = destination_modes[rel]
                    mode = _new_file_mode(parent) if destination_mode is None else destination_mode
                    temp = f".stateframe-{secrets.token_hex(16)}.tmp"
                    descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                         0o600, dir_fd=parent)
                    try:
                        try:
                            temp_identity = _identity(os.fstat(descriptor))
                        except OSError:
                            temp_identity = _identity(_recover_staged_stat(descriptor))
                            staged[rel] = (parent_identity, temp, temp_identity)
                            raise
                        staged[rel] = (parent_identity, temp, temp_identity)
                        # Writes can clear set-ID bits. Set ordinary permissions before writing
                        # and apply any special bits after all content is staged.
                        write_mode = mode & ~(stat.S_ISUID | stat.S_ISGID)
                        os.fchmod(descriptor, write_mode)
                        info = _stat_destination(parent, Path(rel).name)
                        current_mode = stat.S_IMODE(info.st_mode) if info is not None else None
                        if current_mode != destination_mode:
                            raise WorkCopyError("destination mode changed while staging")
                        pending = memoryview(data)
                        while pending:
                            count = os.write(descriptor, pending)
                            if count <= 0: raise OSError("staging write made no progress")
                            pending = pending[count:]
                        if mode != write_mode:
                            os.fchmod(descriptor, mode)
                    finally:
                        os.close(descriptor)
            except (OSError, WorkCopyError) as error:
                raise WorkCopyError(f"{what} staging failed at {rel}: {error}; already applied: nothing") \
                    from error

        conflicts = []
        for rel in sorted(changes):
            parent_context = (_apply_parent_fd(original, rel, staged) if rel in staged
                              else _parent_fd(original, rel))
            try:
                with parent_context as parent:
                    if rel in staged:
                        expected_parent = staged[rel][0]
                        _verify_parent_identity(original, rel, parent, expected_parent)
                    else:
                        checked[rel] = _identity(os.fstat(parent))
                    current, mode = _read_at(parent, Path(rel).name)
                    if rel in staged:
                        _verify_parent_identity(original, rel, parent, expected_parent)
            except FileNotFoundError as error:
                current, mode = None, None
                if rel in staged:
                    conflicts.append(f"{rel} (parent directory changed while applying: {error})")
            except (OSError, WorkCopyError) as error:
                conflicts.append(f"{rel} ({error})")
                continue
            if why := _apply_conflict(current, mode, baseline.get(rel), (alternatives or {}).get(rel)):
                conflicts.append(f"{rel} ({why})")
            elif rel in staged and current != changes[rel] and mode != destination_modes[rel]:
                conflicts.append(f"{rel} (destination mode changed while applying)")
        if conflicts:
            raise WorkCopyError(f"refusing {what}: " + "; ".join(conflicts[:10]))

        for rel in sorted(changes, key=lambda path: (changes[path] is None, path)):
            data, name = changes[rel], Path(rel).name
            try:
                expected_parent = staged[rel][0] if data is not None else checked.get(rel)
                with _apply_parent_fd(original, rel, staged) as parent:
                    _verify_parent_identity(original, rel, parent, expected_parent)
                    current, mode = _read_at(parent, name)
                    why = _apply_conflict(current, mode, baseline.get(rel), (alternatives or {}).get(rel))
                    if why: raise WorkCopyError(why)
                    if current == data:
                        continue
                    _verify_parent_identity(original, rel, parent, expected_parent)
                    if data is None:
                        os.unlink(name, dir_fd=parent)
                    else:
                        _, temp, temp_identity = staged[rel]
                        _verify_staged_identity(parent, temp, temp_identity)
                        # Keep the final leaf read and parent check next to the rename.
                        current, mode = _read_at(parent, name)
                        why = _apply_conflict(current, mode, baseline.get(rel), (alternatives or {}).get(rel))
                        if why:
                            raise WorkCopyError(why)
                        if current == data:
                            continue
                        if mode != destination_modes[rel]:
                            raise WorkCopyError("destination mode changed while applying")
                        _verify_parent_identity(original, rel, parent, expected_parent)
                        os.replace(temp, name, src_dir_fd=parent, dst_dir_fd=parent)
            except (OSError, WorkCopyError) as error:
                raise WorkCopyError(f"{what} stopped at {rel}: {error}; already applied: {applied or 'nothing'}") \
                    from error
            applied.append(rel)
        succeeded = True
        return applied
    finally:
        _remove_staged_temps(original, staged)
        if not succeeded:
            _cleanup_created_dirs(original, created_dirs)


@contextmanager
def _engine_importable():
    """Make the engine's flat kernel_* modules importable for the switch helpers (same pattern as outer_checks)."""
    engine = str(Path(__file__).resolve().parent / "engine")
    added = engine not in sys.path
    if added:
        sys.path.insert(0, engine)
    try:
        yield
    finally:
        if added:
            sys.path.remove(engine)


def _drift_merge_enabled() -> bool:
    """COLLAB on and STRICT off, read by kernel_edit_drift. Any value it rejects keeps today's refusal."""
    with _engine_importable():
        from kernel_edit_drift import drift_merge_enabled
        try:
            return drift_merge_enabled()
        except ValueError:
            return False


def _start_bytes(work: WorkCopy, rel: str) -> bytes | None:
    """Start-of-run bytes of ``rel`` from the immutable snapshot, only if they match the baseline hash."""
    expected = work.baseline.get(rel)
    if work.snapshot is None or expected is None:
        return None
    try:
        with _parent_fd(work.snapshot, rel) as parent:
            data, _ = _read_at(parent, Path(rel).name)
    except (OSError, WorkCopyError):
        return None
    if data is None or _sha(data) != expected:
        return None
    return data


def _line_span(lo: int, hi: int) -> list[int]:
    return [lo + 1, hi] if hi > lo else [lo + 1, lo]


def _line_text(span: tuple[int, int]) -> str:
    first, last = _line_span(*span)
    return f"{first}-{last}"


def _cut_lines(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    if len(lines) <= MAX_HUNK_LINES:
        return text, False
    return "".join(lines[:MAX_HUNK_LINES]), True


def _hunk_payload(hunk: ConflictHunk) -> dict[str, object]:
    base, base_cut = _cut_lines(hunk.base_text)
    run, run_cut = _cut_lines(hunk.ours_text)
    original, original_cut = _cut_lines(hunk.theirs_text)
    return {
        "base_lines": _line_span(*hunk.base_lines),
        "run_lines": _line_span(*hunk.ours_lines),
        "original_lines": _line_span(*hunk.theirs_lines),
        "base_text": base,
        "run_text": run,
        "original_text": original,
        "truncated": base_cut or run_cut or original_cut,
    }


def _merge_outside_edit(base: bytes, run: bytes, original: bytes, rel: str) -> bytes | ConflictReport | None:
    """Merged bytes, a conflict report, or None when the bytes are not mergeable text (today's refusal stands)."""
    if any(b"\x00" in data for data in (base, run, original)):
        return None
    try:
        base_text, run_text, original_text = (data.decode("utf-8") for data in (base, run, original))
    except UnicodeDecodeError:
        return None
    outcome = merge_texts(base_text, run_text, original_text)
    if isinstance(outcome, MergeApplied):
        return outcome.text.encode("utf-8")
    if isinstance(outcome, MergeConflict):
        shown = outcome.hunks[:MAX_HUNKS_PER_FILE]
        spans = ", ".join(f"run lines {_line_text(hunk.ours_lines)} vs original lines {_line_text(hunk.theirs_lines)}"
                          for hunk in shown)
        more = f", +{len(outcome.hunks) - len(shown)} more" if len(outcome.hunks) > len(shown) else ""
        return ConflictReport(path=rel, hunk_count=len(outcome.hunks),
                              hunks=tuple(_hunk_payload(hunk) for hunk in shown), spans=spans + more)
    return None


def _merge_outside_edits(work: WorkCopy, changes: dict[str, bytes | None],
                         baseline: dict[str, str]) -> tuple[dict[str, bytes | None], dict[str, str], list[str]]:
    """Three-way merge each run write whose original file moved since the start (COLLAB-DRIFT on only).

    A merged file's new bytes replace the run's write, and its baseline becomes the original bytes it was merged
    onto, so guarded_apply still rechecks that file right before its own write. Anything not mergeable (deleted or
    created by the user, non-text, no start bytes, ambiguous) is left alone and refused by guarded_apply as before.
    Any real overlap aborts the whole apply with a MergeConflictError, before anything is written.
    """
    merged_changes = dict(changes)
    merged_baseline = dict(baseline)
    merged_paths: list[str] = []
    reports: list[ConflictReport] = []
    for rel, data in list(changes.items()):
        if data is None or baseline.get(rel) is None:
            continue
        try:
            with _parent_fd(work.original, rel) as parent:
                current, _ = _read_at(parent, Path(rel).name)
        except (OSError, WorkCopyError):
            continue
        if current is None or _sha(current) == baseline[rel]:
            continue
        base = _start_bytes(work, rel)
        if base is None:
            continue
        outcome = _merge_outside_edit(base, data, current, rel)
        if outcome is None:
            continue
        if isinstance(outcome, ConflictReport):
            reports.append(outcome)
            continue
        merged_changes[rel] = outcome
        merged_baseline[rel] = _sha(current)
        merged_paths.append(rel)
    if reports:
        parts = [f"{report.path} (merge conflict: {report.spans})" for report in reports[:MAX_CONFLICT_FILES_SHOWN]]
        raise MergeConflictError(
            "refusing --in-place: " + "; ".join(parts) + "; nothing was written", tuple(reports))
    return merged_changes, merged_baseline, merged_paths


def apply_in_place(work: WorkCopy, *, merged: list[str] | None = None,
                   changed: dict[str, str] | None = None,
                   candidate_file_digests: Mapping[str, str] | None = None) -> list[str]:
    """Copy changed files back to the original repository, guarded against concurrent edits.

    With COLLAB-DRIFT on, a file the user edited meanwhile is three-way merged (start snapshot as base) instead of
    refused. Merged paths are appended to ``merged`` only after the whole apply succeeds.
    If supplied, ``changed`` is the host-approved change list for this promotion.
    """
    if changed is None:
        changed = changed_files(work)
    changes: dict[str, bytes | None] = {}
    for rel, kind in changed.items():
        source = work.copy / rel
        if candidate_file_digests is not None and kind == "deleted":
            try:
                # Match the digest scan's view of regular files without following symlinks.
                if (not any((work.copy / part).is_symlink() for part in reversed(Path(rel).parents))
                        and stat.S_ISREG(source.lstat().st_mode)):
                    raise WorkCopyError(f"candidate changed after host verification: {rel}")
            except (FileNotFoundError, NotADirectoryError):
                pass
            except OSError as error:
                raise WorkCopyError(f"candidate changed after host verification: {rel}") from error
            data = None
        else:
            try:
                if kind == "deleted" or candidate_file_digests is None:
                    data = None if kind == "deleted" else source.read_bytes()
                else:
                    with _parent_fd(work.copy, rel) as parent:
                        data, _ = _read_at(parent, source.name)
            except (OSError, WorkCopyError) as error:
                if candidate_file_digests is None:
                    raise
                raise WorkCopyError(f"candidate changed after host verification: {rel}") from error
        if candidate_file_digests is not None and (
                (data is None and rel in candidate_file_digests)
                or (data is not None and candidate_file_digests.get(rel) != _sha(data))):
            raise WorkCopyError(f"candidate changed after host verification: {rel}")
        changes[rel] = data
    baseline: dict[str, str] = work.baseline
    merged_paths: list[str] = []
    if _drift_merge_enabled():
        changes, baseline, merged_paths = _merge_outside_edits(work, changes, baseline)
    _check_apply_space(work.original, changes)
    applied = guarded_apply(work.original, changes, baseline, what="--in-place")
    if merged is not None:
        merged.extend(sorted(merged_paths))
    return applied
