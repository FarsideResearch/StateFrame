"""Retained worker files and exclusive storage ownership, not resume authority."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import stat
import uuid
from pathlib import Path

from kernel_measurement import canonical_bytes, digest

FORMAT = "continuation-workspace/1"


class WorkspaceError(RuntimeError):
    pass


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _outside_protected(path: Path) -> Path:
    value = path.resolve()
    root = Path(__file__).resolve().parent
    for protected in (root / "results", root / "results.SNAPSHOT-20260920"):
        if value == protected or value.is_relative_to(protected):
            raise WorkspaceError("workspace storage cannot use a protected corpus")
    return value


class ContinuationWorkspace:
    """One worker owns this directory; closing never removes its artifacts."""

    @classmethod
    def create(cls, parent: Path, *, task_digest: str) -> ContinuationWorkspace:
        if not isinstance(task_digest, str) or not task_digest.startswith("sha256:") or len(task_digest) != 71:
            raise WorkspaceError("canonical task digest required")
        parent = _outside_protected(Path(parent))
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        identity = uuid.uuid4().hex
        directory = parent / identity
        directory.mkdir(mode=0o700)
        session = cls._acquire(directory, create=True)
        try:
            session._write("allocation.json", {
                "schema": FORMAT, "session_id": identity, "task_digest": task_digest})
            _sync_directory(parent)
        except BaseException:
            session.close()
            raise
        return session

    @classmethod
    def open_existing(cls, directory: Path, *, task_digest: str) -> ContinuationWorkspace:
        """Acquire files for inspection; this does NOT authorize worker execution."""
        original = Path(directory).absolute()
        if original.is_symlink():
            raise WorkspaceError("workspace directory cannot be a symlink")
        session = cls._acquire(_outside_protected(original), create=False)
        try:
            allocation = session.read("allocation.json")
            if (allocation.get("schema") != FORMAT
                    or allocation.get("session_id") != session.directory.name
                    or allocation.get("task_digest") != task_digest):
                raise WorkspaceError("workspace task or identity mismatch")
        except BaseException:
            session.close()
            raise
        return session

    @classmethod
    def _acquire(cls, directory: Path, *, create: bool) -> ContinuationWorkspace:
        value = directory.lstat()
        if not stat.S_ISDIR(value.st_mode):
            raise WorkspaceError("workspace must be a real directory")
        lock = directory / "owner.lock"
        flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        fd = os.open(lock, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise WorkspaceError("workspace lease must be a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(fd)
            raise
        session = cls()
        session.directory, session._fd, session._pid = directory, fd, os.getpid()
        session._identity = (value.st_dev, value.st_ino)
        session._poisoned = False
        return session

    def assert_owner(self) -> None:
        if self._fd is None or self._pid != os.getpid() or self._poisoned:
            raise WorkspaceError("workspace lease is closed, inherited or poisoned")
        directory, lock = self.directory.lstat(), (self.directory / "owner.lock").lstat()
        opened = os.fstat(self._fd)
        if (not stat.S_ISDIR(directory.st_mode)
                or (directory.st_dev, directory.st_ino) != self._identity
                or not stat.S_ISREG(lock.st_mode)
                or (lock.st_dev, lock.st_ino) != (opened.st_dev, opened.st_ino)):
            self._poisoned = True
            raise WorkspaceError("workspace ownership changed")

    def _write(self, name: str, value: dict) -> None:
        self.assert_owner()
        if Path(name).name != name:
            raise WorkspaceError("metadata name must be a direct child")
        data = canonical_bytes(value) + b"\n"
        fd = os.open(self.directory / name,
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            remaining = memoryview(data)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise OSError("incomplete workspace metadata write")
                remaining = remaining[count:]
            os.fsync(fd)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            os.close(fd)
        try:
            _sync_directory(self.directory)
        except BaseException:
            self._poisoned = True
            raise

    def read(self, name: str) -> dict:
        self.assert_owner()
        if Path(name).name != name:
            raise WorkspaceError("metadata name must be a direct child")
        fd = os.open(self.directory / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4 * 1024 * 1024:
                raise WorkspaceError("invalid workspace metadata file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                raw = stream.read(4 * 1024 * 1024 + 1)
        finally:
            os.close(fd)
        value = json.loads(raw)
        if not isinstance(value, dict) or canonical_bytes(value) + b"\n" != raw:
            raise WorkspaceError("metadata is not a canonical object")
        return value

    def bind(self, runtime) -> None:
        self.assert_owner()
        if getattr(runtime, "target", None) is None:
            # Audit 2026-10-05 #3: batches 889/894 died with AttributeError
            # ('NoneType'.resolve) here; fail typed and journaled instead.
            journal = getattr(runtime, "journal", None)
            if journal is not None:
                journal.emit("kernel_error", error="target_unresolved",
                             producer="kernel_continuation_workspace.bind", dispatch=False)
            raise WorkspaceError("retained session requires a resolved target")
        root = Path(runtime.owner.root).resolve()
        target = runtime.target.resolve()
        if root == self.directory or not root.is_relative_to(self.directory) or not target.is_relative_to(root):
            raise WorkspaceError("execution root/target escape the retained session")
        if not root.is_dir():
            raise WorkspaceError("execution workspace is missing")
        initial = {
            "schema": FORMAT, "root": root.relative_to(self.directory).as_posix(),
            "target": target.relative_to(root).as_posix(), "binding": runtime.binding(),
            "protected_tests": {str(p.relative_to(root)): sha for p, sha in runtime.protected_tests.items()},
            "run_id": runtime.journal.run_id, "task_id": runtime.journal.task_id,
            "config_hash": runtime.journal.config_hash, "journal_path": str(runtime.journal.path),
        }
        self._write("initialized.json", initial)
        runtime.journal.emit("continuation_workspace_bound",
            directory=str(self.directory), root=str(root),
            allocation_hash=digest(self.read("allocation.json")), initialization_hash=digest(initial),
            worker_resume_implemented=bool(getattr(runtime, "restart_enabled", False)))

    def grading_copy(self, root: Path) -> Path:
        """Grade a new copy; never mutate the retained execution world."""
        self.assert_owner()
        initial = self.read("initialized.json")
        original = Path(root).resolve()
        bound_root = self.directory / initial["root"]
        if (original == self.directory or not original.is_relative_to(self.directory)
                or bound_root.is_symlink() or original != bound_root.resolve()):
            raise WorkspaceError("grading source is not the bound execution workspace")
        target = self.directory / ("grading-" + uuid.uuid4().hex)
        shutil.copytree(original, target, symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".git"))
        self._write("grading.json", {"source": str(original), "copy": str(target),
                                    "purpose": "terminal_grading_only"})
        return target

    def record_stop(self, journal=None) -> None:
        self._write("stopped.json", {
            "schema": FORMAT, "journal_path": str(journal.path) if journal is not None else None,
            "journal_head": journal.head if journal is not None else None,
            "run_id": journal.run_id if journal is not None else None,
            "execution_authorized": False, "worker_resume_implemented": bool(
                journal is not None and journal.manifest.get("continuation_restart", False))})

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> ContinuationWorkspace:
        self.assert_owner()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
