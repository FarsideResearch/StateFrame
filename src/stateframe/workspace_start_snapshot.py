"""Run-private admission and observations of the freshly materialized worker."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import TYPE_CHECKING

from .workcopy import _READ_FLAGS, _Copies, _Stamp, _stamp

if TYPE_CHECKING:
    from .workcopy import Limits, WorkCopy

_MANIFEST = "parent-start.json"
_MAX_OVERLAY_BYTES = 64 << 20
_CONTEXT_EXCLUSIONS = frozenset({"__pycache__", ".pytest_cache", ".test-tmp"})


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@dataclass
class _Admission:
    staging: tempfile.TemporaryDirectory
    files: dict[str, tuple[str, _Stamp]]
    directories: dict[str, _Stamp]


def admit_workcopy(work: WorkCopy, limits: Limits) -> None:
    """Write both copies from the same descriptor-anchored admission."""
    from .workcopy import (_TEXT_FILE_BYTES, WorkCopyError, _CopySpace, _copy_start_file, _hash_file, _ignored,
                           _is_copy_storage_full, _repository_scan_error, _source_file_error, _walk_source)

    space = _CopySpace(work.copy)

    def scan_error(path: Path, error: OSError) -> None:
        raise _repository_scan_error(path, error) from error

    work.copy.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(prefix=".workspace-start-", dir=work.copy.parent)
    try:
        root = Path(staging.name)
        workspace, parent = root / "workspace", root / "parent"
        workspace.mkdir()
        parent.mkdir()
        copies = work._copies or _Copies(False)
        files: dict[str, tuple[str, _Stamp]] = {}
        directories = {Path(".")}
        for directory, filenames, dir_fd in _walk_source(work.original, onerror=scan_error):
            here = directory.relative_to(work.original)
            for name in filenames:
                relative = here / name
                rel = relative.as_posix()
                if _ignored(name, False):
                    work.note_skipped(rel, "excluded by file name")
                    continue
                path = directory / name
                try:
                    before = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                except OSError as error:
                    raise _source_file_error("inspect", path, error) from error
                if stat.S_ISLNK(before.st_mode):
                    work.note_skipped(rel, "symbolic link")
                    continue
                if not stat.S_ISREG(before.st_mode):
                    work.note_skipped(rel, "special file")
                    continue
                try:
                    source = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
                except OSError as error:
                    if error.errno in {errno.ELOOP, errno.ENXIO}:
                        work.note_skipped(rel, "symbolic link" if error.errno == errno.ELOOP else "special file")
                        continue
                    if isinstance(error, PermissionError):
                        work.note_skipped(rel, "unreadable")
                        continue
                    raise _source_file_error("read", path, error) from error
                with os.fdopen(source, "rb"):
                    info = os.fstat(source)
                    if not stat.S_ISREG(info.st_mode):
                        work.note_skipped(rel, "special file")
                        continue
                    replaced = (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)
                    if replaced and max(info.st_size, before.st_size) <= _TEXT_FILE_BYTES:
                        raise WorkCopyError(f"repository file changed before admission: {rel}")
                    if here not in directories:
                        (workspace / here).mkdir(parents=True, exist_ok=True)
                        (parent / here).mkdir(parents=True, exist_ok=True)
                        directories.update((here, *here.parents))
                    try:
                        copies.mode_only = True
                        copied = _copy_start_file(copies, source, workspace / relative, rel, space)
                        copies.mode_only = False
                    except OSError as error:
                        if _is_copy_storage_full(error):
                            raise
                        raise _source_file_error("copy", path, error) from error
                    if not copied:
                        work.note_skipped(rel, "changed while copying")
                        continue
                    if (workspace / relative).stat().st_size > _TEXT_FILE_BYTES:
                        work.large_files.add(rel)
                    workspace_fd = os.open(workspace / relative, _READ_FLAGS)
                    try:
                        target_info = os.fstat(workspace_fd)
                        digest = _hash_file(workspace / relative)
                        signature = copies.file(workspace_fd, parent / relative, target_info)
                        if _stamp(os.fstat(workspace_fd)) != _stamp(target_info):
                            raise WorkCopyError(f"workspace file changed during snapshot: {rel}")
                    finally:
                        os.close(workspace_fd)
                    files[rel] = (digest, signature)
                    work.baseline[rel] = digest
        directory_stamps = {}
        for directory in sorted(directories, reverse=True):
            shutil.copystat(workspace / directory, parent / directory, follow_symlinks=False)
            directory_stamps[directory.as_posix()] = _stamp((parent / directory).stat())
        if os.path.lexists(work.copy):
            raise WorkCopyError(f"working copy destination already exists: {work.copy}")
        workspace.rename(work.copy)
    except BaseException:
        staging.cleanup()
        raise
    work._admission = _Admission(staging, files, directory_stamps)


def pin_parent(work: WorkCopy, destination: Path) -> Path:
    """Publish the immutable admitted parent. The parent rename commits its sidecar."""
    from .workcopy import WorkCopyError, _CopySpace, _raise_disk_full, copy_tree

    destination = Path(destination).absolute()
    if os.path.lexists(destination):
        raise WorkCopyError(f"parent snapshot destination already exists: {destination}")
    if (destination.parent != work.copy.parent or destination == work.copy
            or destination.resolve().is_relative_to(work.original)):
        raise WorkCopyError("parent snapshot must be a sibling of the working copy")
    admission = work._admission
    if admission is None:
        # A caller may use the same interface for a legacy workcopy.
        try:
            copy_tree(work.copy, destination, work._copies, space=_CopySpace(destination))
        except OSError as error:
            _raise_disk_full(work.original, destination, 1024 * 1024, (destination, work.copy), error)
            raise
    else:
        assert isinstance(admission, _Admission)
        stage = Path(admission.staging.name)
        parent = stage / "parent"
        manifest = destination.parent / _MANIFEST
        manifest_stage = stage / _MANIFEST
        published = False
        payload = {
            "version": 1, "name": destination.name,
            "directories": sorted(admission.directories.items()),
            "files": sorted(admission.files.items(), key=lambda row: tuple(row[0].split("/"))),
        }
        try:
            payload["digest"] = _digest(payload)
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            with open(manifest_stage, "xb", opener=_private_file) as stream:
                stream.write(encoded)
            if os.path.lexists(manifest):
                raise WorkCopyError(f"parent snapshot manifest already exists: {manifest}")
            manifest_stage.rename(manifest)
            published = True
            parent.rename(destination)
        except OSError as error:
            if published:
                manifest.unlink(missing_ok=True)
            _raise_disk_full(work.original, destination, 1024 * 1024, (destination, work.copy), error)
            raise
        finally:
            admission.staging.cleanup()
            work._admission = None
    work.snapshot = destination
    return destination


def _private_file(path: str, flags: int) -> int:
    return os.open(path, flags, 0o600)


def _relative_path(path: object) -> bool:
    return isinstance(path, str) and "\0" not in path and not any(
        part in {"", ".", ".."} for part in path.split("/"))


def _valid_stamp(stamp: object) -> bool:
    return (isinstance(stamp, list) and len(stamp) == 9
            and all(type(value) is int for value in stamp))


def _parent_manifest(parent: Path) -> dict[str, tuple[str, _Stamp]] | None:
    try:
        fd = os.open(parent.parent / _MANIFEST, _READ_FLAGS)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077 or info.st_nlink != 1
                    or info.st_size <= 0):
                return None
            data = stream.read()
            if len(data) != info.st_size or _stamp(os.fstat(fd)) != _stamp(info):
                return None
            payload = json.loads(data)
        if (not isinstance(payload, dict) or payload["version"] != 1
                or payload["name"] != parent.name or payload.pop("digest") != _digest(payload)):
            return None
        directories = {}
        for path, stamp in payload["directories"]:
            if ((path != "." and not _relative_path(path)) or path in directories
                    or not _valid_stamp(stamp) or not stat.S_ISDIR(stamp[2])):
                return None
            directories[path] = tuple(stamp)
        # Every ancestor was admitted. A changed directory can change the
        # manifest's membership or introduce an alias even if file bytes match.
        for path, stamp in sorted(directories.items(), key=lambda row: Path(row[0]).parts):
            if path != "." and Path(path).parent.as_posix() not in directories:
                return None
            actual = _stamp((parent / path).lstat())
            if path == ".":
                # The commit rename changes the root directory's ctime.
                actual = (*actual[:5], stamp[5], *actual[6:])
            if actual != stamp:
                return None
        if "." not in directories:
            return None
        files = {}
        for path, (digest, stamp) in payload["files"]:
            if (not _relative_path(path) or Path(path).parent.as_posix() not in directories
                    or path in files or not _valid_stamp(stamp)
                    or not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or not stat.S_ISREG(stamp[2]) or stamp[3] < 0):
                return None
            files[path] = (digest, tuple(stamp))
        if not files or list(files) != sorted(files, key=lambda path: tuple(path.split("/"))):
            return None
        return files
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError):
        return None


@dataclass(frozen=True, slots=True)
class _WorkerContext:
    root: Path
    copied: dict[str, tuple[str, _Stamp]]
    declared: tuple[str, ...]
    public_tests: tuple[str, ...]


def materialize_worker(parent: Path, root: Path, fileset: object, row: object) -> _WorkerContext | None:
    """Populate the private worker around the engine's already written declared files."""
    manifest = _parent_manifest(parent)
    if manifest is None:
        return None
    # The engine's public-test materializer runs after this function. Existing
    # declared files take precedence, including new files absent in the parent.
    from .workcopy import WorkCopyError, _parent_fd

    copies = _Copies()
    copied: dict[str, tuple[str, _Stamp]] = {}
    directories = {root}
    for relative, (digest, expected) in manifest.items():
        path = Path(relative)
        if _CONTEXT_EXCLUSIONS.intersection(path.parts):
            continue
        target = root / path
        if target.parent not in directories:
            with _parent_fd(root, relative, create=True):
                directories.add(target.parent)
        try:
            target_mode = target.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(target_mode):
                raise WorkCopyError(f"worker context destination is a symlink: {relative}")
            continue
        source = os.open(parent / path, _READ_FLAGS)
        try:
            info = os.fstat(source)
            if _stamp(info) != expected:
                raise WorkCopyError(f"parent snapshot changed before worker materialization: {relative}")
            signature = copies.file(source, target, info, writable=True)
            if _stamp(os.fstat(source)) != expected:
                raise WorkCopyError(f"parent snapshot changed during worker materialization: {relative}")
        finally:
            os.close(source)
        copied[relative] = (digest, signature)
    return _WorkerContext(
        root, copied, tuple(sorted(getattr(fileset, "parent", ()))),
        tuple(sorted(getattr(fileset, "public_tests", ()))))


def _inventory(root: Path) -> dict[str, _Stamp]:
    """Complete no-follow membership, including directories and their modes."""
    result = {"": _stamp(root.lstat())}
    if not stat.S_ISDIR(result[""][2]):
        raise OSError("worker root is no longer a directory")
    pending = [(os.fspath(root), "")]
    device = result[""][0]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = prefix + entry.name
                info = entry.stat(follow_symlinks=False)
                result[path] = _stamp(info)
                if info.st_dev != device:
                    raise OSError("worker inventory exceeds observation scope")
                if stat.S_ISDIR(info.st_mode):
                    pending.append((entry.path, path + "/"))
                elif not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise OSError("worker inventory includes an unsupported file type or alias")
    return result


@dataclass(frozen=True, slots=True)
class StartupObservation:
    root: Path
    purpose: str
    identity: str
    membership_digest: str
    files: Mapping[str, str]  # Relative POSIX path to hex SHA-256, in path order.
    modes: Mapping[str, int]  # Full stat mode, including types and the root.
    stamps: Mapping[str, _Stamp]
    overlay_digest: str
    exclusion_digest: str

    def validate_for_use(self, purpose: str) -> bool:
        if purpose != self.purpose:
            return False
        current = startup_observation(self.root, purpose)
        return current is not None and current.identity == self.identity


@dataclass(frozen=True, slots=True)
class _WorkerGeneration:
    root: Path
    pid: int
    inventory: dict[str, _Stamp]
    files: Mapping[str, str]
    modes: Mapping[str, int]
    membership_digest: str
    overlay_digest: str
    exclusion_digest: str


_worker: _WorkerGeneration | None = None
_worker_revoked = False
_worker_lock = RLock()


def finalize_worker(context: _WorkerContext, written: tuple[str, ...]) -> None:
    """Capture the effective worker after the original public-test overlay."""
    global _worker

    if _worker_revoked:
        return
    root, copied = context.root, context.copied
    try:
        inventory = _inventory(root)
        files, overlays = {}, {}
        total = 0
        modes = {}
        for relative, signature in sorted(inventory.items(), key=lambda row: tuple(row[0].split("/"))):
            mode = signature[2]
            modes[relative] = mode
            if stat.S_ISDIR(mode):
                if mode & 0o500 != 0o500:
                    return
                continue
            if not mode & stat.S_IRUSR:
                return
            known = copied.get(relative)
            if known is not None and signature == known[1]:
                files[relative] = known[0]
                continue
            total += signature[3]
            if total > _MAX_OVERLAY_BYTES:
                return
            fd = os.open(root / relative, _READ_FLAGS)
            with os.fdopen(fd, "rb") as stream:
                if _stamp(os.fstat(fd)) != signature:
                    return
                data = stream.read(signature[3] + 1)
                if len(data) != signature[3] or _stamp(os.fstat(fd)) != signature:
                    return
            files[relative] = overlays[relative] = hashlib.sha256(data).hexdigest()
        generation = _WorkerGeneration(
            root, os.getpid(), inventory, MappingProxyType(files), MappingProxyType(modes),
            _digest([(relative, mode, files.get(relative)) for relative, mode in modes.items()]),
            _digest([context.declared, context.public_tests, written, list(overlays.items())]),
            _digest(sorted(_CONTEXT_EXCLUSIONS)))
    except OSError:
        # The legacy consumers own the per-path error and link semantics.
        return
    with _worker_lock:
        if not _worker_revoked:
            _worker = generation


def revoke_startup_observation(root: Path | None = None) -> None:
    """End this attempt's reuse before effects with an opaque write footprint."""
    global _worker, _worker_revoked

    with _worker_lock:
        if root is None or (_worker is not None and _worker.root == Path(root).absolute()):
            _worker = None
            _worker_revoked = True


def startup_observation(root: Path, purpose: str) -> StartupObservation | None:
    """Use admitted bytes only while the complete initial generation still matches."""
    with _worker_lock:
        worker = _worker
    if (worker is None or worker.pid != os.getpid()
            or Path(root).absolute() != worker.root or not purpose):
        return None
    try:
        unchanged = _inventory(worker.root) == worker.inventory
    except OSError:
        unchanged = False
    with _worker_lock:
        if _worker is not worker:
            return None
        if not unchanged:
            revoke_startup_observation()
            return None
        identity = _digest([
            1, os.fspath(worker.root), worker.pid, worker.inventory[""], worker.membership_digest,
            worker.overlay_digest, worker.exclusion_digest, purpose,
        ])
        return StartupObservation(
            worker.root, purpose, identity, worker.membership_digest, worker.files, worker.modes,
            MappingProxyType(worker.inventory), worker.overlay_digest, worker.exclusion_digest)
