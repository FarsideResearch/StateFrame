"""Content-addressed, no-follow workspace effect accounting and restoration."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Iterator, Literal, Mapping, Protocol, TextIO


DISCOVERY_SCOPE_REFUSAL = "discovery scope is empty; no source path is admitted for writing"
MAX_FILE_BYTES = 16 << 20
_FICLONE = 0x40049409
_CLONE_UNSUPPORTED = {errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.ENOTTY, errno.EOPNOTSUPP}


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    # Links and their times may change while a shared object is read; its content does not.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)


def _verify_fd(fd: int, digest: str) -> os.stat_result:
    before = os.fstat(fd)
    if not stat.S_ISREG(before.st_mode):
        raise EffectError("invalid CAS file type")
    actual, total = hashlib.sha256(), 0
    while block := os.read(fd, 1 << 20):
        total += len(block)
        actual.update(block)
    if total != before.st_size or actual.hexdigest() != digest or _file_identity(os.fstat(fd)) != _file_identity(before):
        raise EffectError(f"CAS object {digest} is corrupt")
    os.lseek(fd, 0, os.SEEK_SET)
    return before


class EffectError(RuntimeError):
    """The workspace cannot be inspected or its pre-call state cannot be trusted."""


class EffectRestoreError(EffectError):
    def __init__(self, paths: list[str], details: list[str] | None = None):
        self.paths = tuple(sorted(set(paths)))
        message = "workspace restoration failed for: " + ", ".join(self.paths)
        if details:
            message += " (" + "; ".join(details) + ")"
        super().__init__(message)


class ContentStore:
    """Immutable SHA-256 objects in ``run_dir/cas/aa/rest-of-digest``."""

    def __init__(self, run_dir: Path):
        self.root = Path(run_dir) / "cas"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise EffectError(f"invalid CAS digest: {digest!r}")
        return self.root / digest[:2] / digest[2:]

    def put(self, data: bytes) -> str:
        if len(data) > MAX_FILE_BYTES:
            raise EffectError("CAS file exceeds per-file byte limit")
        digest = hashlib.sha256(data).hexdigest()
        target = self._path(digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if self.get(digest) != data:
                raise EffectError(f"CAS object does not match digest {digest}")
            return digest
        fd, temporary = tempfile.mkstemp(prefix=".cas-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o444)
            # The store is private to this run; replacing an identical concurrent object is safe.
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return digest

    def get(self, digest: str) -> bytes:
        path = self._path(digest)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as source:
                if os.fstat(source.fileno()).st_size > MAX_FILE_BYTES:
                    raise EffectError("CAS file exceeds per-file byte limit")
                data = source.read(MAX_FILE_BYTES + 1)
                if len(data) > MAX_FILE_BYTES:
                    raise EffectError("CAS file exceeds per-file byte limit")
        except OSError as error:
            raise EffectError(f"CAS object {digest} unavailable: {error}") from error
        if hashlib.sha256(data).hexdigest() != digest:
            raise EffectError(f"CAS object {digest} is corrupt")
        return data

    def verify(self, digest: str) -> None:
        """Verify an existing CAS object with bounded memory and no link following."""
        fd = os.open(self._path(digest), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            _verify_fd(fd, digest)
        finally:
            os.close(fd)

    def copy_to(self, digest: str, output: BinaryIO) -> None:
        fd = os.open(self._path(digest), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = _verify_fd(fd, digest)
            try:
                fcntl.ioctl(output.fileno(), _FICLONE, fd)
                output.flush()
            except (AttributeError, OSError) as error:
                if isinstance(error, OSError) and error.errno is not None and error.errno not in _CLONE_UNSUPPORTED:
                    raise
                output.seek(0)
                output.truncate()
                while block := os.read(fd, 1 << 20):
                    if output.write(block) != len(block):
                        raise OSError("short copy to restored file")
            if _file_identity(os.fstat(fd)) != _file_identity(before):
                raise EffectError(f"CAS object {digest} changed while copying")
        finally:
            os.close(fd)

    def capture(self, workspace: Path, manifest: Manifest) -> None:
        """Seed pre-call bytes; reject a workspace changed since its manifest was scanned."""
        _validate_manifest_limits(manifest)
        for relative, entry in manifest.items():
            if entry.kind != "file":
                continue
            with _capture_source(workspace, relative, entry) as fd:
                target = self._path(entry.digest)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    # Known object: the live source still gets the same drift checks,
                    # but no temp file, fsync or link is needed.
                    _stream_source(fd, relative, entry, None)
                    # Verification is bounded, independently of the workspace.
                    self.verify(entry.digest)
                    continue
                out_fd, temporary = tempfile.mkstemp(prefix=".cas-", dir=target.parent)
                try:
                    with os.fdopen(out_fd, "wb") as output:
                        _stream_source(fd, relative, entry, output)
                        output.flush()
                        os.fsync(output.fileno())
                    if target.exists():
                        # Object appeared during the copy; verify it instead of replacing it.
                        self.verify(entry.digest)
                    else:
                        os.chmod(temporary, 0o444)
                        os.link(temporary, target)
                finally:
                    os.unlink(temporary)


class BaselineStore:
    """The pinned rollback files for one manifest, with legacy CAS read support."""

    def __init__(self, directory: Path, manifest: Manifest):
        _validate_manifest_limits(manifest)
        self._directory = directory
        self._root = directory / manifest_digest(manifest)
        self._index: dict[str, list[int]] | None = None
        self._layout: dict[str, list[int]] = {}
        self._files: list[tuple[str, Entry, bool]] = []
        self._pack_size = 0
        for relative, entry in sorted(manifest.items()):
            if entry.kind != "file":
                continue
            digest = entry.digest
            if digest is None:
                raise EffectError("baseline file has no digest")
            first = digest not in self._layout
            self._files.append((relative, entry, first))
            if first:
                # Equal bytes share a slice. The manifest still owns each path's mode.
                self._layout[digest] = [self._pack_size, entry.size, entry.mode]
                self._pack_size += entry.size

    def publish(self, workspace: Path) -> None:
        if (index := self._read_index()) is None:
            self._directory.mkdir(mode=0o700, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".baseline-", dir=self._directory) as name:
                staging = Path(name)
                self._write_pack(workspace, staging / "objects.pack")
                with (staging / "index.json").open("w", encoding="utf-8") as output:
                    json.dump(self._layout, output, separators=(",", ":"))
                    _sync_baseline_file(output)
                _sync_baseline_directory(staging)
                # A directory rename publishes both files together, after they are durable.
                try:
                    os.replace(staging, self._root)
                except OSError as error:
                    if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                        raise
                    index = self._read_index()
                    if index is None:
                        raise
                else:
                    index = self._layout
        # Also finish the directory barriers if an earlier attempt stopped after the rename.
        _sync_baseline_directory(self._directory)
        _sync_baseline_directory(self._directory.parent)
        self._index = index

    def _write_pack(self, workspace: Path, pack: Path) -> None:
        with pack.open("wb") as output:
            for relative, entry, first in self._files:
                with _capture_source(workspace, relative, entry, check_mode=True) as fd:
                    _stream_source(fd, relative, entry, output if first else None)
            _sync_baseline_file(output)

    def _read_index(self) -> dict[str, list[int]] | None:
        try:
            info = self._root.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISDIR(info.st_mode):
            raise EffectError("invalid rollback baseline root")
        try:
            # The compact representation bounds the index read even if on-disk data is damaged.
            limit = 2 + 128 * len(self._layout)
            with _open_baseline_file(self._root / "index.json") as source:
                data = source.read(limit + 1)
                if len(data) > limit:
                    raise EffectError("rollback baseline index exceeds its size limit")
            index = json.loads(data)
            with _open_baseline_file(self._root / "objects.pack") as pack:
                if os.fstat(pack.fileno()).st_size != self._pack_size:
                    raise EffectError("rollback baseline pack has an invalid size")
        except (OSError, ValueError, RecursionError) as error:
            raise EffectError(f"rollback baseline is unreadable: {error}") from error
        if index != self._layout or any(
            type(number) is not int for record in index.values() for number in record
        ):
            raise EffectError("rollback baseline index does not match its manifest")
        return index

    def get(self, digest: str) -> bytes:
        if self._index is None:
            self._index = self._read_index()
        if self._index is None or digest not in self._index:
            return ContentStore(self._directory).get(digest)
        offset, length, _ = self._index[digest]
        try:
            with _open_baseline_file(self._root / "objects.pack") as pack:
                pack.seek(offset)
                data = pack.read(length)
        except OSError as error:
            raise EffectError(f"rollback baseline object {digest} unavailable: {error}") from error
        if len(data) != length or hashlib.sha256(data).hexdigest() != digest:
            raise EffectError(f"rollback baseline object {digest} is corrupt")
        return data

    def copy_to(self, digest: str, output: BinaryIO) -> None:
        if self._index is None:
            self._index = self._read_index()
        if self._index is None or digest not in self._index:
            ContentStore(self._directory).copy_to(digest, output)
            return
        offset, length, _ = self._index[digest]
        with _open_baseline_file(self._root / "objects.pack") as pack:
            pack.seek(offset)
            remaining, actual = length, hashlib.sha256()
            while remaining:
                block = pack.read(min(1 << 20, remaining))
                if not block:
                    raise EffectError(f"rollback baseline object {digest} is corrupt")
                remaining -= len(block)
                actual.update(block)
                output.write(block)
        if actual.hexdigest() != digest:
            raise EffectError(f"rollback baseline object {digest} is corrupt")


@contextmanager
def _open_baseline_file(path: Path) -> Iterator[BinaryIO]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise EffectError("invalid rollback baseline file type")
        yield source


def _sync_baseline_file(output: BinaryIO | TextIO) -> None:
    output.flush()
    fd = output.fileno()
    os.fchmod(fd, 0o444)
    os.fsync(fd)


def _sync_baseline_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True, slots=True)
class Entry:
    kind: Literal["file", "symlink", "dir", "other"]
    mode: int
    digest: str | None
    size: int


Manifest = Mapping[str, Entry]


@contextmanager
def _capture_source(
    workspace: Path, relative: str, entry: Entry, *, check_mode: bool = False
) -> Iterator[int]:
    fd = os.open(Path(workspace) / relative, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size != entry.size
            or (check_mode and stat.S_IMODE(info.st_mode) != entry.mode)
        ):
            raise EffectError(f"workspace changed while capturing {relative}")
        yield fd
    finally:
        os.close(fd)


def _stream_source(
    fd: int, relative: str, entry: Entry, output: BinaryIO | None
) -> None:
    """Hash an open workspace file against its manifest entry, copying it when ``output`` is set.

    Raises the same messages, in the same order, as the copying capture always did.
    """
    digest = hashlib.sha256()
    total = 0
    before = os.fstat(fd)
    while block := os.read(fd, 1 << 20):
        total += len(block)
        if total > entry.size:
            raise EffectError(f"workspace grew while capturing {relative}")
        digest.update(block)
        if output is not None:
            output.write(block)
    if (total != entry.size or digest.hexdigest() != entry.digest
            or _file_identity(os.fstat(fd)) != _file_identity(before)):
        raise EffectError(f"workspace changed while capturing {relative}")


def _read_file(path: Path) -> bytes:
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise EffectError(f"not a regular unlinked file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise EffectError(f"file changed while opening: {path}")
        with os.fdopen(fd, "rb", closefd=False) as source:
            if opened.st_size > MAX_FILE_BYTES:
                raise EffectError(f"file exceeds per-file byte limit: {path}")
            data = source.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise EffectError(f"file exceeds per-file byte limit: {path}")
            return data
    finally:
        os.close(fd)


def _validate_manifest_limits(manifest: Manifest) -> None:
    if any(entry.kind == "file" and entry.size < 0
           for entry in manifest.values()):
        raise EffectError("workspace file has an invalid size")


def scan(workspace: Path, *, exclude: Callable[[str], bool]) -> Manifest:
    """Hash every regular file; classify hardlinks as unsupported special entries."""
    root = Path(workspace)
    if not stat.S_ISDIR(os.lstat(root).st_mode):
        raise EffectError(f"workspace is not a directory: {root}")
    result: dict[str, Entry] = {}

    def visit(directory: Path) -> None:
        with os.scandir(directory) as children:
            for child in children:
                relative = (directory / child.name).relative_to(root).as_posix()
                if exclude(relative):
                    continue
                info = child.stat(follow_symlinks=False)
                mode = stat.S_IMODE(info.st_mode)
                if stat.S_ISDIR(info.st_mode):
                    result[relative] = Entry("dir", mode, None, 0)
                    visit(Path(child.path))
                elif stat.S_ISLNK(info.st_mode):
                    link = os.readlink(child.path).encode(
                        "utf-8", errors="surrogateescape"
                    )
                    result[relative] = Entry(
                        "symlink", mode, hashlib.sha256(link).hexdigest(), len(link)
                    )
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    digest = hashlib.sha256()
                    fd = os.open(child.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                    try:
                        opened = os.fstat(fd)
                        if _file_identity(opened) != _file_identity(info):
                            raise EffectError(
                                f"workspace changed while scanning {relative}"
                            )
                        consumed = 0
                        while block := os.read(fd, 1 << 20):
                            consumed += len(block)
                            if consumed > info.st_size:
                                raise EffectError(f"workspace grew while scanning {relative}")
                            digest.update(block)
                        if (consumed != info.st_size
                                or _file_identity(os.fstat(fd)) != _file_identity(opened)):
                            raise EffectError(
                                f"workspace changed while scanning {relative}"
                            )
                    finally:
                        os.close(fd)
                    result[relative] = Entry(
                        "file", mode, digest.hexdigest(), opened.st_size
                    )
                else:
                    result[relative] = Entry("other", mode, None, info.st_size)

    visit(root)
    return result


@dataclass(frozen=True)
class Effect:
    path: str
    change: Literal["created", "modified", "deleted", "mode", "type"]
    before: Entry | None
    after: Entry | None


def classify_effect(path: str, old: Entry | None, new: Entry | None) -> Effect | None:
    if old == new:
        return None
    if old is None:
        change = "created"
    elif new is None:
        change = "deleted"
    elif old.kind != new.kind:
        change = "type"
    elif old.digest != new.digest or old.size != new.size:
        change = "modified"
    else:
        change = "mode"
    return Effect(path, change, old, new)


def diff(before: Manifest, after: Manifest) -> list[Effect]:
    effects = []
    for path in sorted(before.keys() | after.keys()):
        effect = classify_effect(path, before.get(path), after.get(path))
        if effect is not None:
            effects.append(effect)
    return effects


class EffectReader(Protocol):
    def get(self, digest: str) -> bytes: ...
    def copy_to(self, digest: str, output: BinaryIO) -> None: ...


class EffectPolicy(Protocol):
    def authorize(self, effect: Effect) -> tuple[bool, str]: ...


BAD_NAME_REASON = "file names must be valid UTF-8"


@dataclass(frozen=True)
class WorkspaceEffectPolicy:
    protected: Callable[[str], bool]
    edit_scope: Callable[[str], bool] | None
    new_file_allowed: Callable[[str], bool]
    baseline_manifest: Manifest | None = None
    directory_allowed: Callable[[str], bool] | None = None
    collection_guard: Callable[[str, bool, str | None, str | None], str | None] | None = None
    # Files a task's explicit file-scope rule allows ("Change no other file"); None when no rule.
    task_scope: tuple[str, ...] | None = None
    edit_scope_refusal: str = "outside edit scope"

    def authorize(self, effect: Effect) -> tuple[bool, str]:
        if self.protected(effect.path):
            return False, "protected test or inode alias"
        if self.collection_guard is not None and effect.before is None:
            if reason := self.collection_guard(effect.path, True, None, None):
                return False, reason
        if effect.after and effect.after.kind in {"symlink", "other"}:
            return False, "symlinks, hardlinks, and special files are not allowed"
        if effect.after and not effect.path.isascii():
            try:
                effect.path.encode("utf-8")
            except UnicodeEncodeError:
                return False, BAD_NAME_REASON
        if (
            effect.before
            and effect.before.kind == "file"
            and effect.before.mode & 0o111
            and effect.after is None
        ):
            return (
                False,
                "shell removal of executable files cannot preserve move metadata; use file_delete",
            )
        if effect.after and effect.after.kind == "file":
            previous = (
                effect.before.mode & 0o111
                if effect.before and effect.before.kind == "file"
                else 0
            )
            if effect.after.mode & 0o111 != previous:
                return (
                    False,
                    "executable mode changes are not supported by the sealed artifact",
                )
        if effect.before is None and effect.after and effect.after.kind == "dir":
            allowed = self.directory_allowed is not None and self.directory_allowed(
                effect.path
            )
            return (
                allowed,
                ("structural product directory" if allowed else
                 self.edit_scope_refusal if self.edit_scope_refusal != "outside edit scope" else
                 "new directory is not allowed"),
            )
        # The task's file-scope rule is a hard limit and is checked before the generic edit scope.
        if self.task_scope is not None and effect.path not in self.task_scope:
            return False, "outside task file scope"
        if self.edit_scope is not None and not self.edit_scope(effect.path):
            return False, self.edit_scope_refusal
        if effect.before is None and not self.new_file_allowed(effect.path):
            return False, "new path is not allowed"
        return True, "authorized workspace effect"


def _workspace_relative(root: Path, path: object) -> str:
    """A baseline test path as workspace-relative POSIX text.

    Engine callers pass absolute paths under the workspace; projected worker rows carry paths that are
    already workspace-relative (``tests/test_app.py``). ``relative_to`` raises on the latter, so a relative
    path is kept as it is and an absolute one is made relative, resolving symlinks if the plain form fails.
    """
    candidate = Path(str(path))
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.resolve().relative_to(root.resolve()).as_posix()


def effect_policy_for_run(
    workspace: Path,
    row: Mapping[str, object],
    baseline_tests: set[str],
    *,
    baseline_manifest: Manifest | None = None,
    hook_paths: frozenset[str] = frozenset(),
) -> WorkspaceEffectPolicy:
    """Adapt the engine's own path decisions; keep their policy in one place."""
    from probes.react_repair import _is_product_path, protected_test_path
    from stateframe.test_runners import collection_change_reason, pytest_plugin_modules

    root = Path(workspace)
    files = row.get("files")
    if not isinstance(files, Mapping):
        raise EffectError("run row has no product files mapping")
    editable = frozenset(str(path) for path in files)
    named_scope = row.get("task_file_scope")
    task_scope = (tuple(str(path) for path in named_scope)
                  if isinstance(named_scope, (list, tuple)) and named_scope else None)
    original_paths = frozenset(
        (baseline_manifest or scan(root, exclude=lambda _: False)).keys()
    )
    guarded = (row.get("runner_kind", "pytest") == "pytest"
               and row.get("protect_tests", "all") in {"task", "all"})
    test_paths = tuple(str(path) for path in {
        *(row.get("test_file_paths", ()) or ()),
        *(row.get("public_check_paths", ()) or ())})
    protected_paths = tuple(_workspace_relative(root, path) for path in baseline_tests)
    plugins = (pytest_plugin_modules(root, tuple(
        path for path in original_paths if path.endswith(".py")
        and (path in test_paths or Path(path).name in {"conftest.py", "__init__.py"})))
        if guarded else ())

    def collection_guard(path: str, created: bool, before: str | None,
                         after: str | None) -> str | None:
        current_plugins = plugins
        if created and path.endswith(".py"):
            sources = tuple(candidate.relative_to(root).as_posix()
                            for pattern in ("test_*.py", "conftest.py", "__init__.py")
                            for candidate in root.rglob(pattern))
            current_plugins += pytest_plugin_modules(root, sources)
        return collection_change_reason(
            path, kind="pytest", protected_tests=protected_paths,
            test_files=test_paths, pytest_plugins=current_plugins, created=created,
            before=before, after=after)
    protected_inodes = set()
    for name in baseline_tests:
        try:
            info = os.lstat(name)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if stat.S_ISREG(info.st_mode):
            protected_inodes.add((info.st_dev, info.st_ino))
    for name in hook_paths:
        try:
            info = os.lstat(root / name)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if stat.S_ISREG(info.st_mode):
            # Hardlinks have no content digest in a workspace scan. An alias
            # present before the first action could mutate a protected file
            # without producing an effect to restore.
            if info.st_nlink != 1:
                raise EffectError(f"protected hook file has hardlinks: {name}")
            protected_inodes.add((info.st_dev, info.st_ino))

    def is_protected(path: str) -> bool:
        if path in hook_paths:
            return True
        candidate = root / path
        if (guarded and path in original_paths
                and collection_guard(path, False, None, None)):
            return True
        if protected_test_path(candidate, baseline_tests):
            return True
        try:
            info = os.lstat(candidate)
        except (FileNotFoundError, NotADirectoryError):
            return False
        if hook_paths and stat.S_ISREG(info.st_mode):
            for name in hook_paths:
                try:
                    protected = os.lstat(root / name)
                except (FileNotFoundError, NotADirectoryError):
                    continue
                if stat.S_ISREG(protected.st_mode) and (info.st_dev, info.st_ino) == (
                    protected.st_dev,
                    protected.st_ino,
                ):
                    return True
        return (info.st_dev, info.st_ino) in protected_inodes

    discovery_scope = row.get("scope_source") == "discovery"
    return WorkspaceEffectPolicy(
        protected=is_protected,
        edit_scope=(
            None
            if row.get("scoped_edits") is False and not discovery_scope
            else lambda path: (
                path in editable
                # Files admitted by opt-in edit scope widening during the run
                # (react_repair EDIT_SCOPE_WIDENED_KEY); absent unless widened.
                or path in row.get("_edit_scope_widened", ())
                or (not discovery_scope
                    and path not in original_paths and _is_product_path(path, row))
            )
        ),
        new_file_allowed=lambda path: (
            not discovery_scope and _is_product_path(path, row)),
        collection_guard=collection_guard if guarded else None,
        baseline_manifest=baseline_manifest,
        task_scope=task_scope,
        edit_scope_refusal=(DISCOVERY_SCOPE_REFUSAL if discovery_scope else "outside edit scope"),
        directory_allowed=lambda path: (
            not discovery_scope and ((
                row.get("tier") == "USER-REPO"
                and row.get("scoped_edits") is False
                and _is_product_path(path, row)
            )
            or any(
                path == str(base).rstrip("/")
                or path.startswith(str(base).rstrip("/") + "/")
                for base in row.get("allowed_product_paths", ())
                if (root / str(base)).is_dir()
            ))
        ),
    )


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {".", ".."} for part in pure.parts)
    ):
        raise EffectError(f"invalid workspace path: {relative!r}")
    current = root
    for part in pure.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise EffectError(f"symlink ancestor while restoring {relative}")
    return root.joinpath(*pure.parts)


def _remove(path: Path) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode):
        with os.scandir(path) as children:
            for child in children:
                _remove(Path(child.path))
        os.rmdir(path)
    else:
        os.unlink(path)


def _restore(
    root: Path,
    relative: str,
    before: Manifest,
    store: EffectReader,
    baseline_store: EffectReader | None,
    policy: EffectPolicy,
) -> None:
    target = _safe_path(root, relative)
    old = before.get(relative)
    if old is None:
        _remove(target)
        return
    if old.kind not in {"file", "dir"}:
        raise EffectError(f"cannot restore pre-call {old.kind} at {relative}")
    if old.kind == "dir":
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            _remove(target)
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, old.mode | stat.S_IWUSR | stat.S_IXUSR)
        with os.scandir(target) as children:
            for child in children:
                child_relative = f"{relative}/{child.name}"
                if child_relative not in before:
                    _remove(Path(child.path))
        descendants = sorted(
            (p for p in before if p.startswith(relative + "/")),
            key=lambda p: (p.count("/"), p),
        )
        for child in descendants:
            _restore(root, child, before, store, baseline_store, policy)
        os.chmod(target, old.mode)
        return
    if old.digest is None:
        raise EffectError(f"pre-call file has no digest: {relative}")
    use_baseline = (
        isinstance(policy, WorkspaceEffectPolicy)
        and policy.baseline_manifest is not None
        and policy.protected(relative)
    )
    if use_baseline and baseline_store is None:
        raise EffectError(f"protected baseline store unavailable for {relative}")
    source = baseline_store if use_baseline else store
    baseline = (
        policy.baseline_manifest.get(relative)
        if isinstance(policy, WorkspaceEffectPolicy) and policy.baseline_manifest
        else None
    )
    entry = baseline if use_baseline and baseline is not None else old
    if entry.digest is None:
        raise EffectError(f"baseline file has no digest: {relative}")
    copy_to = getattr(source, "copy_to", None)
    stream = copy_to is not None and entry.size > MAX_FILE_BYTES
    data = b"" if stream else source.get(entry.digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".effect-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            if stream:
                copy_to(entry.digest, output)
            else:
                output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, entry.mode)
        if target.is_symlink() or (target.exists() and target.is_dir()):
            _remove(target)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class Reconciliation:
    applied: list[Effect]
    restored: list[tuple[Effect, str]]
    manifest: Manifest


def reconcile(
    workspace: Path,
    before: Manifest,
    after: Manifest,
    policy: EffectPolicy,
    store: ContentStore,
    baseline_store: ContentStore | BaselineStore | None,
) -> Reconciliation:
    _validate_manifest_limits(before)
    _validate_manifest_limits(after)

    def observe_final() -> Manifest:
        included = before.keys() | after.keys()
        prefixes = {
            str(parent)
            for path in included
            for parent in PurePosixPath(path).parents
            if str(parent) != "."
        }
        return scan(
            Path(workspace),
            exclude=lambda path: path not in included and path not in prefixes,
        )

    result = reconcile_effects(
        workspace, before, after, policy, store, baseline_store,
        changes=diff(before, after), observe_final=observe_final,
    )
    store.capture(Path(workspace), result.manifest)
    return result


def reconcile_effects(
    workspace: Path,
    before: Manifest,
    after: Manifest,
    policy: EffectPolicy,
    store: EffectReader,
    baseline_store: EffectReader | None,
    *,
    changes: Iterable[Effect],
    observe_final: Callable[[], Manifest],
    restore_scope: Callable[[list[str]], AbstractContextManager[None]] = lambda _: nullcontext(),
) -> Reconciliation:
    """Apply legacy effect policy and verify compensation before publication."""
    decisions = []
    for effect in changes:
        decision = policy.authorize(effect)
        if (decision[0] and isinstance(policy, WorkspaceEffectPolicy)
                and policy.collection_guard is not None
                and effect.before is not None and effect.before.kind == "file"
                and (effect.after is None or effect.after.kind == "file")
                and PurePosixPath(effect.path).name in {
                    "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml"}):
            if effect.before.size > MAX_FILE_BYTES or effect.after and effect.after.size > MAX_FILE_BYTES:
                decision = False, "large test configuration changes are not allowed"
            else:
                original = store.get(effect.before.digest).decode("utf-8", errors="replace")
                current = ((Path(workspace) / effect.path).read_text(
                    encoding="utf-8", errors="replace") if effect.after is not None else "")
                if reason := policy.collection_guard(effect.path, False, original, current):
                    decision = False, reason
        decisions.append((effect, *decision))
    # A rename to a name that is restored must not lose its source.
    lost = {effect.after.digest for effect, allowed, reason in decisions
            if not allowed and reason == BAD_NAME_REASON and effect.after.digest}
    # Directories have no digest: an emptied one stays when a new directory was refused its name.
    lost_dir = any(not allowed and reason == BAD_NAME_REASON and effect.before is None
                   and effect.after is not None and effect.after.kind == "dir"
                   for effect, allowed, reason in decisions)
    parents = {str(parent) for path in before for parent in PurePosixPath(path).parents} if lost_dir else set()
    decisions = [(effect, False, BAD_NAME_REASON)
                 if allowed and effect.after is None and (
                     effect.before.digest in lost
                     or lost_dir and effect.before.kind == "dir" and effect.path not in parents)
                 else (effect, allowed, reason) for effect, allowed, reason in decisions]
    # A new directory is structural authority, not a product-file extension.
    # Keep it when an admitted descendant needs it, but never override protection.
    if isinstance(policy, WorkspaceEffectPolicy):
        admitted_parents = {
            str(parent) for effect, allowed, _ in decisions if allowed
            for parent in PurePosixPath(effect.path).parents}
        decisions = [
            (effect, True, "parent of authorized effect")
            if (
                not allowed
                and effect.before is None
                and effect.after is not None
                and effect.after.kind == "dir"
                and not policy.protected(effect.path)
                and effect.path in admitted_parents
            )
            else (effect, allowed, reason)
            for effect, allowed, reason in decisions
        ]
    denied = [(effect, reason) for effect, allowed, reason in decisions if not allowed]
    # Only restore an ancestor when its current type prevents reaching the child,
    # or that ancestor is itself denied. Restoring an authorized new directory
    # would discard authorized siblings of a denied child.
    denied_paths = {effect.path for effect, _ in denied}
    roots = []
    root_paths = set()
    for effect, _ in sorted(
        denied, key=lambda item: (item[0].path.count("/"), item[0].path)
    ):
        candidate = effect.path
        for parent in PurePosixPath(effect.path).parents:
            ancestor = str(parent)
            if ancestor in denied_paths or (
                ancestor in before
                and before[ancestor].kind == "dir"
                and (ancestor not in after or after[ancestor].kind != "dir")
            ):
                candidate = str(parent)
        if candidate not in root_paths and not any(
            str(parent) in root_paths for parent in PurePosixPath(candidate).parents
        ):
            roots.append(candidate)
            root_paths.add(candidate)
    failures = []
    failure_details = []
    if roots:
        with restore_scope(roots):
            for path in roots:
                try:
                    _restore(Path(workspace), path, before, store, baseline_store, policy)
                except (OSError, EffectError) as error:
                    failures.append(path)
                    failure_details.append(f"{path}: {error}")
    if failures:
        raise EffectRestoreError(failures, failure_details)
    final = observe_final()
    for effect, _ in denied:
        expected = before.get(effect.path)
        if (
            isinstance(policy, WorkspaceEffectPolicy)
            and policy.baseline_manifest
            and policy.protected(effect.path)
        ):
            expected = policy.baseline_manifest.get(effect.path, expected)
        if final.get(effect.path) != expected:
            failures.append(effect.path)
    if failures:
        raise EffectRestoreError(failures)
    applied = []
    for effect, allowed, _ in decisions:
        if allowed:
            if final.get(effect.path) == effect.after:
                applied.append(effect)
            else:
                denied.append((effect, "parent restoration also removed this effect"))
    return Reconciliation(
        applied=applied,
        restored=denied,
        manifest=final,
    )


def manifest_digest(manifest: Manifest) -> str:
    payload = [(path, asdict(entry)) for path, entry in sorted(manifest.items())]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
